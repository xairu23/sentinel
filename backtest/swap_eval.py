"""Test a variant of v3 where one or both history-based families are swapped for a variant cache
(all-53 replay), on the primary score (log WIS, all 53 places, vs the CDC baseline) with the round-3
rule. Used for ideas 6, 18 and 19 in backtest/V3_PROTOCOL.md.

    python -m backtest.swap_eval --tree gbqsp --label "18. trees + neighbouring states"
    python -m backtest.swap_eval --tree gbqbf --qar qarbf --label "19. corrected newest week for trees and QAR"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.eval_all53 import HUB, load_family, wis_rows
from backtest.exp_v3 import SEASONS, quantile_average
from backtest.leaderboard import load_model
from backtest.member_cache import mean_members
from backtest.round3 import no_pr
from backtest.run_backtest import final_hosp

ROOT = Path(__file__).resolve().parents[1]


def run(tree: str = "gbq", qar: str = "qar", label: str = "variant", base_tree: str = "gbq") -> str:
    """base_tree: the trees of the version compared against (gbq = v3, gbqbf = v3.1)."""
    mem, g0, q0 = load_family("mem"), load_family(base_tree), load_family("qar")
    g1 = load_family(tree)
    q1 = load_family(qar) if qar != "none" else q0
    refs = sorted(set(mem) & set(g0) & set(q0) & set(g1) & set(q1))
    season = {r: mem[r]["season"] for r in refs}
    lin = {r: mean_members(mem[r]["mem"]) for r in refs}
    v3 = {r: quantile_average([lin[r], g0[r][base_tree], no_pr(q0[r]["qar"])]) for r in refs}
    if qar == "none":  # idea 20c: no QAR at all
        new = {r: quantile_average([lin[r], g1[r][tree]]) for r in refs}
    else:
        new = {r: quantile_average([lin[r], g1[r][tree], no_pr(q1[r][qar])]) for r in refs}
    truth = final_hosp().set_index(["location", "date"])["value"]
    rows = []
    for r in refs:
        last = r - pd.Timedelta(weeks=1)
        for (loc, k) in v3[r]:
            y = truth.get((loc, last + pd.Timedelta(weeks=k)))
            if y is not None and not np.isnan(y) and (loc, k) in new[r]:
                rows.append((season[r], r, str(r.date()), loc, k - 1, float(y)))
    keys = pd.DataFrame(rows, columns=["season", "ref", "ref_s", "location", "horizon", "y"])
    idx = pd.MultiIndex.from_frame(keys[["ref_s", "location", "horizon"]])
    w = load_model(HUB / "model-output/FluSight-baseline", keys.rename(columns={"ref_s": "ref"})[["ref", "location", "horizon"]])
    w.index = w.index.set_names(["ref_s", "location", "horizon"])
    base = w.reindex(idx).values.astype(float)
    ok = ~np.isnan(base).any(axis=1)
    keys, base, y = keys[ok].reset_index(drop=True), base[ok], keys["y"].values[ok]
    lg = lambda a: np.log1p(np.maximum(a, 0))

    def sc(p):
        Q = np.round(np.vstack([p[r][(l, h + 1)] for r, l, h in zip(keys.ref, keys.location, keys.horizon)]))
        return wis_rows(lg(Q), lg(y)), wis_rows(Q, y), Q

    W0, C0, Q0 = sc(v3)
    W1, C1, Q1 = sc(new)
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
    above0, above1 = (y > Q0[:, 11]).mean(), (y > Q1[:, 11]).mean()
    line = (f"| {label} | {W1.sum() / Wb.sum():.3f} (v3 {W0.sum() / Wb.sum():.3f}) | "
            f"{(B.sum() / A.sum() - 1) * 100:+.1f}% ({lo:+.1f} to {hi:+.1f}) | "
            + " | ".join(f"{ps[s]:+.1f}%" for s in SEASONS)
            + f" | {cnt:+.1f}% | {above0:.0%} -> {above1:.0%} | {'KEEP' if keep else 'fails'} |")
    print("| idea | score | change vs v3 (95% CI) | " + " | ".join(SEASONS)
          + " | counts, 50 states | truth above median | rule |")
    print(line)
    return line


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default="gbq")
    ap.add_argument("--qar", default="qar")
    ap.add_argument("--label", default="variant")
    ap.add_argument("--base-tree", default="gbq", help="gbq (v3) or gbqbf (v3.1)")
    a = ap.parse_args()
    run(a.tree, a.qar, a.label, a.base_tree)
