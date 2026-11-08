"""Round 3 decision script (backtest/V3_PROTOCOL.md, ideas 11 and 12), on the primary score:
WIS of log(x+1), all 53 places, the exact live setup (all-53 replay caches), relative to the CDC
baseline. Change vs v3 with a 95% interval from resampling whole weeks; per-season changes; and the
50-state count score as the guard-rail.

    python -m backtest.round3
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backtest.eval_all53 import HUB, load_family, wis_rows
from backtest.exp_v3 import SEASONS, quantile_average
from backtest.leaderboard import load_model
from backtest.member_cache import mean_members
from backtest.run_backtest import final_hosp
from models.engine import V3

ROOT = Path(__file__).resolve().parents[1]


def no_pr(q: dict) -> dict:
    return {k: v for k, v in q.items() if k[0] not in V3.qar_skip}


def candidates() -> tuple[dict, dict]:
    fam = {f: load_family(f) for f in ("mem", "gbq", "qar", "ar6", "qarw")}
    refs = sorted(set.intersection(*(set(v) for v in fam.values())))
    season = {r: fam["mem"][r]["season"] for r in refs}
    lin = {r: mean_members(fam["mem"][r]["mem"]) for r in refs}
    g = {r: fam["gbq"][r]["gbq"] for r in refs}
    q = {r: no_pr(fam["qar"][r]["qar"]) for r in refs}
    qw = {r: no_pr(fam["qarw"][r]["qarw"]) for r in refs}
    a = {r: fam["ar6"][r]["ar6"] for r in refs}
    c = {
        "v3": {r: quantile_average([lin[r], g[r], q[r]]) for r in refs},
        "11. v3 + AR6 (quarters)": {r: quantile_average([lin[r], g[r], q[r], a[r]]) for r in refs},
        "12. v3 with QAR weighted to NHSN": {r: quantile_average([lin[r], g[r], qw[r]]) for r in refs},
        "11+12": {r: quantile_average([lin[r], g[r], qw[r], a[r]]) for r in refs},
        "AR6 alone": a, "QAR alone": {r: fam["qar"][r]["qar"] for r in refs},
        "QAR weighted alone": {r: fam["qarw"][r]["qarw"] for r in refs},
    }
    return c, season


def main():
    c, season = candidates()
    refs = sorted(season)
    truth = final_hosp().set_index(["location", "date"])["value"]
    rows = []
    for r in refs:
        last = r - pd.Timedelta(weeks=1)
        for (loc, k) in c["v3"][r]:
            y = truth.get((loc, last + pd.Timedelta(weeks=k)))
            if y is not None and not np.isnan(y):
                rows.append((season[r], r, str(r.date()), loc, k - 1, float(y)))
    keys = pd.DataFrame(rows, columns=["season", "ref", "ref_s", "location", "horizon", "y"])
    idx = pd.MultiIndex.from_frame(keys[["ref_s", "location", "horizon"]])
    w = load_model(HUB / "model-output/FluSight-baseline", keys.rename(columns={"ref_s": "ref"})[["ref", "location", "horizon"]])
    w.index = w.index.set_names(["ref_s", "location", "horizon"])
    base = w.reindex(idx).values.astype(float)
    ok = ~np.isnan(base).any(axis=1)
    keys, base, y = keys[ok].reset_index(drop=True), base[ok], keys["y"].values[ok]
    lg = lambda a: np.log1p(np.maximum(a, 0))
    W = {"base": wis_rows(lg(base), lg(y))}
    Wc = {"base": wis_rows(base, y)}
    for name, p in c.items():
        Q = np.round(np.vstack([p[r].get((l, h + 1), np.full(23, np.nan)) for r, l, h in
                                zip(keys.ref, keys.location, keys.horizon)]))
        W[name], Wc[name] = wis_rows(lg(Q), lg(y)), wis_rows(Q, y)
    states = ~keys["location"].isin(["US", "11", "72"]).values
    rng = np.random.default_rng(0)
    wk = keys["ref"].values
    weeks = np.unique(wk)
    lines = ["# Round 3: decision on the CDC's ranking scale (log, all 53 places)", "",
             f"{len(keys):,} forecasts, {len(weeks)} weeks. Relative WIS vs the CDC baseline; change vs v3.", "",
             "| candidate | log, all 53, vs CDC baseline | change vs v3 (95% CI) | "
             + " | ".join(SEASONS) + " | counts, 50 states: change vs v3 | rule |", "|---" * 8 + "|"]
    for name in c:
        rel = W[name].sum() / W["base"].sum()
        if name == "v3":
            lines.append(f"| v3 | {rel:.3f} | - | " + " | ".join("-" for _ in SEASONS) + " | - | |")
            continue
        if np.isnan(W[name]).any():
            lines.append(f"| {name} | (missing forecasts) |")
            continue
        A = pd.Series(W["v3"]).groupby(wk).sum().reindex(weeks).values
        B = pd.Series(W[name]).groupby(wk).sum().reindex(weeks).values
        ii = rng.integers(0, len(weeks), size=(2000, len(weeks)))
        boots = (B[ii].sum(1) / A[ii].sum(1) - 1) * 100
        lo, hi = np.percentile(boots, [2.5, 97.5])
        pt = (B.sum() / A.sum() - 1) * 100
        ps = {s: (W[name][(keys.season == s).values].sum() / W["v3"][(keys.season == s).values].sum() - 1) * 100
              for s in SEASONS}
        cnt = (Wc[name][states].sum() / Wc["v3"][states].sum() - 1) * 100
        keep = hi < 0 and all(v <= 0 for v in ps.values()) and cnt <= 1.0
        family_only = "alone" in name
        lines.append(f"| {name} | {rel:.3f} | {pt:+.1f}% ({lo:+.1f} to {hi:+.1f}) | "
                     + " | ".join(f"{ps[s]:+.1f}%" for s in SEASONS)
                     + f" | {cnt:+.1f}% | {'(reference)' if family_only else ('KEEP' if keep else 'fails')} |")
    lines += idea13(c["v3"], keys, y, W, Wc, states)
    out = "\n".join(lines)
    (ROOT / "backtest/round3.md").write_text(out + "\n")
    print(out)


def shrink(pred: dict, c: float, thresh: float = 30.0) -> dict:
    """Idea 13: pull every quantile towards the median by factor c when the median is below thresh."""
    out = {}
    for key, q in pred.items():
        if q[11] < thresh and c != 1.0:
            q = np.maximum.accumulate(np.maximum(q[11] + c * (q - q[11]), 0))
        out[key] = q
    return out


def idea13(v3: dict, keys: pd.DataFrame, y: np.ndarray, W: dict, Wc: dict, states: np.ndarray) -> list:
    lg = lambda a: np.log1p(np.maximum(a, 0))
    grid = (0.6, 0.7, 0.8, 0.9, 1.0)
    Wl, Wn = {}, {}
    for cf in grid:
        Q = np.round(np.vstack([shrink({(l, h + 1): v3[r][(l, h + 1)]}, cf)[(l, h + 1)]
                                for r, l, h in zip(keys.ref, keys.location, keys.horizon)]))
        Wl[cf], Wn[cf] = wis_rows(lg(Q), lg(y)), wis_rows(Q, y)
    new, newc, chosen = W["v3"].copy(), Wc["v3"].copy(), {}
    for s in SEASONS:
        test = (keys.season == s).values
        best = min(grid, key=lambda cf: Wl[cf][~test].sum())
        chosen[s] = best
        new[test], newc[test] = Wl[best][test], Wn[best][test]
    wk = keys["ref"].values
    weeks = np.unique(wk)
    A = pd.Series(W["v3"]).groupby(wk).sum().reindex(weeks).values
    B = pd.Series(new).groupby(wk).sum().reindex(weeks).values
    ii = np.random.default_rng(0).integers(0, len(weeks), size=(2000, len(weeks)))
    lo, hi = np.percentile((B[ii].sum(1) / A[ii].sum(1) - 1) * 100, [2.5, 97.5])
    pt = (B.sum() / A.sum() - 1) * 100
    ps = {s: (new[(keys.season == s).values].sum() / W["v3"][(keys.season == s).values].sum() - 1) * 100 for s in SEASONS}
    cnt = (newc[states].sum() / Wc["v3"][states].sum() - 1) * 100
    keep = hi < 0 and all(v <= 0 for v in ps.values()) and cnt <= 1.0
    return [f"| 13. narrower ranges below 30 (LOSO; chosen {chosen}) | {new.sum() / W['base'].sum():.3f} | "
            f"{pt:+.1f}% ({lo:+.1f} to {hi:+.1f}) | " + " | ".join(f"{ps[s]:+.1f}%" for s in SEASONS)
            + f" | {cnt:+.1f}% | {'KEEP' if keep else 'fails'} |"]


if __name__ == "__main__":
    main()
