"""Round 4 (backtest/V3_PROTOCOL.md, ideas 14 and 15): correcting v3's tendency to forecast too low.

14: in-season bias correction learned from the last 6 weeks of v3's own forecasts that have come true,
    compared with the data AS PUBLISHED at forecast time; strength s chosen leave-one-season-out.
15: reference: a fixed log shift per horizon, chosen leave-one-season-out.
Primary score: WIS of log(x+1), all 53 places, vs the CDC baseline (same keys as backtest/round3.py).

    python -m backtest.round4
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backtest.eval_all53 import HUB, wis_rows
from backtest.exp_v3 import SEASONS
from backtest.leaderboard import load_model
from backtest.round3 import candidates
from backtest.run_backtest import Vintages, final_hosp

ROOT = Path(__file__).resolve().parents[1]
WINDOW, MIN_WEEKS = 6, 3


def inseason_bias(v3: dict, refs: list, V: Vintages) -> dict:
    """{(ref, k): b}: median log((published + 1) / (median + 1)) of v3's resolved forecasts of horizon k
    made in the last WINDOW weeks, against the version published for this week."""
    out = {}
    for r in refs:
        last = r - pd.Timedelta(weeks=1)
        as_of = V.max_date[V.max_date == last].index.min()
        pub = V.t[V.t["as_of"] == as_of].set_index(["location", "date"])["value"]
        for k in (1, 2, 3, 4):
            vals, weeks = [], set()
            for r2 in refs:
                if not (r - pd.Timedelta(weeks=WINDOW) <= r2 < r):
                    continue
                t = r2 - pd.Timedelta(weeks=1) + pd.Timedelta(weeks=k)
                if t > last:
                    continue
                for (loc, kk), q in v3[r2].items():
                    if kk != k:
                        continue
                    y = pub.get((loc, t))
                    if y is not None and not np.isnan(y):
                        vals.append(np.log((y + 1) / (q[11] + 1)))
                        weeks.add(r2)
            out[(r, k)] = float(np.median(vals)) if len(weeks) >= MIN_WEEKS else 0.0
    return out


def main():
    c, season = candidates()
    v3 = c["v3"]
    refs = sorted(season)
    V = Vintages(HUB)
    bias = inseason_bias(v3, refs, V)
    truth = final_hosp().set_index(["location", "date"])["value"]
    rows = []
    for r in refs:
        last = r - pd.Timedelta(weeks=1)
        for (loc, k) in v3[r]:
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
    Q0 = np.vstack([v3[r][(l, h + 1)] for r, l, h in zip(keys.ref, keys.location, keys.horizon)])
    b = np.array([bias[(r, h + 1)] for r, h in zip(keys.ref, keys.horizon)])
    states = ~keys["location"].isin(["US", "11", "72"]).values

    def score(Q):
        Q = np.round(np.maximum.accumulate(np.maximum(Q, 0), axis=1))
        return wis_rows(lg(Q), lg(y)), wis_rows(Q, y)

    W0, C0 = score(Q0)
    Wb = wis_rows(lg(base), lg(y))

    def loso(cands: dict) -> tuple[np.ndarray, np.ndarray, dict]:
        new, newc, chosen = W0.copy(), C0.copy(), {}
        for s in SEASONS:
            test = (keys.season == s).values
            best = min(cands, key=lambda p: cands[p][0][~test].sum())
            chosen[s] = best
            new[test], newc[test] = cands[best][0][test], cands[best][1][test]
        return new, newc, chosen

    def report(name, new, newc, chosen):
        wk = keys["ref"].values
        weeks = np.unique(wk)
        A = pd.Series(W0).groupby(wk).sum().reindex(weeks).values
        B = pd.Series(new).groupby(wk).sum().reindex(weeks).values
        ii = np.random.default_rng(0).integers(0, len(weeks), size=(2000, len(weeks)))
        lo, hi = np.percentile((B[ii].sum(1) / A[ii].sum(1) - 1) * 100, [2.5, 97.5])
        pt = (B.sum() / A.sum() - 1) * 100
        ps = {s: (new[(keys.season == s).values].sum() / W0[(keys.season == s).values].sum() - 1) * 100 for s in SEASONS}
        cnt = (newc[states].sum() / C0[states].sum() - 1) * 100
        keep = hi < 0 and all(v <= 0 for v in ps.values()) and cnt <= 1.0
        return (f"| {name} | {new.sum() / Wb.sum():.3f} | {pt:+.1f}% ({lo:+.1f} to {hi:+.1f}) | "
                + " | ".join(f"{ps[s]:+.1f}%" for s in SEASONS) + f" | {cnt:+.1f}% | {chosen} | "
                + ("KEEP" if keep else "fails") + " |")

    lines = ["# Round 4: correcting v3's low bias (log scale, all 53 places, vs the CDC baseline)", "",
             f"v3: {W0.sum() / Wb.sum():.3f}. In-season bias b, mean by season: "
             + ", ".join(f"{s} {b[(keys.season == s).values].mean():+.3f}" for s in SEASONS), "",
             "| idea | score | change vs v3 (95% CI) | " + " | ".join(SEASONS) + " | counts, 50 states | chosen | rule |",
             "|---" * 9 + "|"]
    c14 = {s_: score(Q0 * np.exp(s_ * b)[:, None]) for s_ in (0.0, 0.25, 0.5, 0.75, 1.0)}
    lines.append(report("14. in-season bias correction", *loso(c14)))
    kk = keys["horizon"].values
    c15 = {}
    for sh in (-0.05, 0.0, 0.05, 0.10, 0.15, 0.20):
        for hz in range(4):
            pass
    # 15: one shift per horizon, chosen per horizon LOSO
    new, newc, chosen = W0.copy(), C0.copy(), {}
    grid = (-0.05, 0.0, 0.05, 0.10, 0.15, 0.20)
    sc = {sh: score(Q0 * np.exp(sh)) for sh in grid}
    for s in SEASONS:
        test = (keys.season == s).values
        for hz in range(4):
            m = kk == hz
            best = min(grid, key=lambda sh: sc[sh][0][~test & m].sum())
            chosen[(s, hz)] = best
            new[test & m], newc[test & m] = sc[best][0][test & m], sc[best][1][test & m]
    lines.append(report("15. fixed shift per horizon (reference)", new, newc, chosen))
    out = "\n".join(lines)
    (ROOT / "backtest/round4.md").write_text(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
