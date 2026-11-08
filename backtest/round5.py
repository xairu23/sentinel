"""Idea 16 (backtest/V3_PROTOCOL.md): last week's CDC FluSight-ensemble as a fourth voice.

For a round with reference date r, the CDC ensemble published for r - 7 days (made the Thursday
after the previous deadline, so available by this Wednesday) forecast the weeks r, r+7 and r+14 as
its horizons 1-3. Those are this round's horizons 0-2. They're averaged in with weight 1/4.

    python -m backtest.round5
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backtest.eval_all53 import HUB, wis_rows
from backtest.exp_v3 import SEASONS
from backtest.leaderboard import load_model
from backtest.round3 import candidates
from backtest.run_backtest import final_hosp
from backtest.wis import QUANTILES

ROOT = Path(__file__).resolve().parents[1]
QSTR = [f"{q:g}" for q in QUANTILES]


def stale_ensemble(refs: list) -> dict:
    """{ref: {(loc, k): q}} from the CDC ensemble file of ref - 7 days (k = 1..3 only)."""
    out = {}
    for r in refs:
        f = HUB / "model-output/FluSight-ensemble" / f"{(r - pd.Timedelta(days=7)).date()}-FluSight-ensemble.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f, dtype={"location": str, "output_type_id": str})
        d = d[(d["target"] == "wk inc flu hosp") & (d["output_type"] == "quantile") & d["horizon"].between(1, 3)]
        d["output_type_id"] = d["output_type_id"].astype(float).map(lambda x: f"{x:g}")
        w = d.pivot_table(index=["location", "horizon"], columns="output_type_id", values="value")
        if not set(QSTR) <= set(w.columns):
            continue
        out[r] = {(loc, int(h)): np.maximum.accumulate(row[QSTR].values.astype(float))
                  for (loc, h), row in w.iterrows()}  # its horizon h = our k (h-1 + 1)
    return out


def main():
    c, season = candidates()
    v3 = c["v3"]
    refs = sorted(season)
    st = stale_ensemble(refs)
    # v3 is itself the mean of 3 families, so the stale ensemble gets a quarter: (3 * v3 + stale) / 4
    cand = {r: {key: (np.mean([v3[r][key]] * 3 + [st[r][key]], axis=0) if r in st and key in st[r] else q)
                for key, q in v3[r].items()} for r in refs}
    truth = final_hosp().set_index(["location", "date"])["value"]
    rows = []
    for r in refs:
        last = r - pd.Timedelta(weeks=1)
        for (loc, k) in v3[r]:
            y = truth.get((loc, last + pd.Timedelta(weeks=k)))
            if y is not None and not np.isnan(y):
                rows.append((season[r], r, str(r.date()), loc, k - 1, float(y), r in st and (loc, k) in st[r]))
    keys = pd.DataFrame(rows, columns=["season", "ref", "ref_s", "location", "horizon", "y", "has_stale"])
    idx = pd.MultiIndex.from_frame(keys[["ref_s", "location", "horizon"]])
    w = load_model(HUB / "model-output/FluSight-baseline", keys.rename(columns={"ref_s": "ref"})[["ref", "location", "horizon"]])
    w.index = w.index.set_names(["ref_s", "location", "horizon"])
    base = w.reindex(idx).values.astype(float)
    ok = ~np.isnan(base).any(axis=1)
    keys, base, y = keys[ok].reset_index(drop=True), base[ok], keys["y"].values[ok]
    lg = lambda a: np.log1p(np.maximum(a, 0))

    def sc(p):
        Q = np.round(np.maximum.accumulate(np.maximum(np.vstack(
            [p[r][(l, h + 1)] for r, l, h in zip(keys.ref, keys.location, keys.horizon)]), 0), axis=1))
        return wis_rows(lg(Q), lg(y)), wis_rows(Q, y)

    W0, C0 = sc(v3)
    W1, C1 = sc(cand)
    Wb = wis_rows(lg(base), lg(y))
    wk = keys["ref"].values
    weeks = np.unique(wk)
    A = pd.Series(W0).groupby(wk).sum().reindex(weeks).values
    B = pd.Series(W1).groupby(wk).sum().reindex(weeks).values
    ii = np.random.default_rng(0).integers(0, len(weeks), size=(2000, len(weeks)))
    lo, hi = np.percentile((B[ii].sum(1) / A[ii].sum(1) - 1) * 100, [2.5, 97.5])
    ps = {s: (W1[(keys.season == s).values].sum() / W0[(keys.season == s).values].sum() - 1) * 100 for s in SEASONS}
    states = ~keys["location"].isin(["US", "11", "72"]).values
    cnt = (C1[states].sum() / C0[states].sum() - 1) * 100
    keep = hi < 0 and all(v <= 0 for v in ps.values()) and cnt <= 1.0
    byh = {h: round((W1[keys.horizon == h].sum() / W0[keys.horizon == h].sum() - 1) * 100, 1) for h in range(4)}
    lines = ["# Idea 16: last week's CDC ensemble as a fourth voice (log scale, all 53 places)", "",
             f"Forecasts with a stale ensemble available: {keys.has_stale.mean():.0%}.", "",
             f"| | score | change vs v3 (95% CI) | " + " | ".join(SEASONS) + " | counts, 50 states | rule |",
             "|---" * 7 + "|",
             f"| v3 | {W0.sum() / Wb.sum():.3f} | - | - | - | - | - | |",
             f"| 16. + last week's CDC ensemble | {W1.sum() / Wb.sum():.3f} | {(B.sum() / A.sum() - 1) * 100:+.1f}% "
             f"({lo:+.1f} to {hi:+.1f}) | " + " | ".join(f"{ps[s]:+.1f}%" for s in SEASONS)
             + f" | {cnt:+.1f}% | {'KEEP' if keep else 'fails'} |", "",
             f"By horizon (change vs v3): {byh}"]
    out = "\n".join(lines)
    (ROOT / "backtest/round5.md").write_text(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
