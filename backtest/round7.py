"""Ideas 18 (neighbouring states for the trees) and 6 (weeks from Christmas for the trees), each on
its own against v3: the tree family is swapped for the variant, everything else unchanged.
Primary score and rule as rounds 3-6.

    python -m backtest.exp_v3 --produce-all gbqsp gbqx
    python -m backtest.round7
"""
from __future__ import annotations

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
VARIANTS = {"gbqsp": "18. trees + neighbouring states", "gbqx": "6. trees + weeks from Christmas"}


def main():
    mem, gbq, qar = load_family("mem"), load_family("gbq"), load_family("qar")
    alts = {f: load_family(f) for f in VARIANTS}
    refs = sorted(set(mem) & set(gbq) & set(qar) & set.intersection(*(set(a) for a in alts.values())))
    season = {r: mem[r]["season"] for r in refs}
    lin = {r: mean_members(mem[r]["mem"]) for r in refs}
    preds = {"v3": {r: quantile_average([lin[r], gbq[r]["gbq"], no_pr(qar[r]["qar"])]) for r in refs},
             "trees alone": {r: gbq[r]["gbq"] for r in refs}}
    for f, label in VARIANTS.items():
        preds[label] = {r: quantile_average([lin[r], alts[f][r][f], no_pr(qar[r]["qar"])]) for r in refs}
        preds[label + " (trees alone)"] = {r: alts[f][r][f] for r in refs}
    truth = final_hosp().set_index(["location", "date"])["value"]
    rows = []
    for r in refs:
        last = r - pd.Timedelta(weeks=1)
        for (loc, k) in preds["v3"][r]:
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
    Wb = wis_rows(lg(base), lg(y))
    W, C = {}, {}
    for name, p in preds.items():
        Q = np.round(np.vstack([p[r][(l, h + 1)] for r, l, h in zip(keys.ref, keys.location, keys.horizon)]))
        W[name], C[name] = wis_rows(lg(Q), lg(y)), wis_rows(Q, y)
    wk = keys["ref"].values
    weeks = np.unique(wk)
    states = ~keys["location"].isin(["US", "11", "72"]).values
    lines = ["# Round 7: tree-model variants (log scale, all 53 places, vs the CDC baseline)", "",
             "| | score | change vs v3 (95% CI) | " + " | ".join(SEASONS) + " | counts, 50 states | rule |",
             "|---" * 7 + "|"]
    for name in preds:
        rel = W[name].sum() / Wb.sum()
        if name == "v3":
            lines.append(f"| v3 | {rel:.3f} | - | - | - | - | - | |")
            continue
        ref_name = "trees alone" if name.endswith("(trees alone)") else "v3"
        A = pd.Series(W[ref_name]).groupby(wk).sum().reindex(weeks).values
        B = pd.Series(W[name]).groupby(wk).sum().reindex(weeks).values
        ii = np.random.default_rng(0).integers(0, len(weeks), size=(2000, len(weeks)))
        lo, hi = np.percentile((B[ii].sum(1) / A[ii].sum(1) - 1) * 100, [2.5, 97.5])
        ps = {s: (W[name][(keys.season == s).values].sum() / W[ref_name][(keys.season == s).values].sum() - 1) * 100
              for s in SEASONS}
        cnt = (C[name][states].sum() / C[ref_name][states].sum() - 1) * 100
        keep = hi < 0 and all(v <= 0 for v in ps.values()) and cnt <= 1.0
        verdict = "(vs trees alone)" if ref_name != "v3" else ("KEEP" if keep else "fails")
        lines.append(f"| {name} | {rel:.3f} | {(B.sum() / A.sum() - 1) * 100:+.1f}% ({lo:+.1f} to {hi:+.1f}) | "
                     + " | ".join(f"{ps[s]:+.1f}%" for s in SEASONS) + f" | {cnt:+.1f}% | {verdict} |")
    out = "\n".join(lines)
    (ROOT / "backtest/round7.md").write_text(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
