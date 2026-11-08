"""Leaderboard the way the CDC ranks FluSight teams: pairwise relative WIS, on counts AND on
log-transformed counts.

The hub says final rankings are "relative WIS using log-transformed data" (model-output/README.md,
"Evaluation criteria"), with relative WIS computed by the pairwise method of Cramer et al. (PNAS 2022):
for every pair of models, the ratio of their mean WIS on the forecasts both made; a model's skill is
the geometric mean of its ratios against all other models, divided by the baseline's. On the log
scale (WIS of log(x+1) forecasts against log(y+1)) every state counts about equally and errors are
relative, so small states matter as much as California.

Sentinel's quantiles come from the replay caches (data as published each week; see
backtest/exp_v3.py), rounded to whole numbers as they would be submitted. One Sentinel version enters
the tournament at a time (v2, then v3), so they don't affect each other's scores.

    python -m backtest.leaderboard_official [--hub ../FluSight-forecast-hub]
Output: backtest/leaderboard_official.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.exp_v3 import Book, blend, quantile_average
from backtest.leaderboard import load_model
from backtest.member_cache import mean_members
from backtest.wis import QUANTILES

ROOT = Path(__file__).resolve().parents[1]


def wis_rows(Q: np.ndarray, y: np.ndarray) -> np.ndarray:
    ind = (y[:, None] <= Q).astype(float)
    return 2.0 * (((ind - QUANTILES[None, :]) * (Q - y[:, None])).mean(axis=1))


def pairwise(W: np.ndarray, names: list, base: str) -> pd.Series:
    """W: (forecasts, models) WIS with NaN where a model has no forecast. Cramer et al. 2022."""
    M = W.shape[1]
    ok = ~np.isnan(W)
    theta = np.full((M, M), np.nan)
    for i in range(M):
        for j in range(M):
            m = ok[:, i] & ok[:, j]
            if m.sum() and W[m, j].sum() > 0:
                theta[i, j] = W[m, i].sum() / W[m, j].sum()
    skill = np.exp(np.nanmean(np.log(theta), axis=1))
    s = pd.Series(skill, index=names)
    return s / s[base]


def main(hub: Path, out: Path):
    book = Book()
    keys = book.keys.copy()
    keys["ref_s"] = keys["ref"].dt.date.astype(str)
    keys["horizon"] = keys["k"] - 1
    idx = pd.MultiIndex.from_frame(keys[["ref_s", "location", "horizon"]])
    y = keys["y"].values

    lin = {r: mean_members(book.mem[r]["members"]) for r in book.refs}
    ours = {"Sentinel-v2 (replay)": {r: blend(lin[r], book.gbq[r], 0.33) for r in book.refs},
            "Sentinel-v3 (replay)": {r: quantile_average([lin[r], book.gbq[r], book.qar[r]]) for r in book.refs}}
    Qs = {n: np.round(np.vstack([p[r][(l, k)] for r, l, k in zip(keys.ref, keys.location, keys.k)]))
          for n, p in ours.items()}

    # every hub model, on the same keys
    hub_q = {}
    folders = sorted(p for p in (hub / "model-output").iterdir() if p.is_dir())
    kdf = keys.rename(columns={"ref_s": "ref"})[["ref", "location", "horizon"]]
    for i, folder in enumerate(folders):
        w = load_model(folder, kdf)
        if w is None or w.empty:
            continue
        w.index = w.index.set_names(["ref_s", "location", "horizon"])
        w = w[w.index.isin(idx)]
        if len(w):
            hub_q[folder.name] = w.reindex(idx).values.astype(float)
        print(f"{i + 1}/{len(folders)} {folder.name}", flush=True)

    lines = ["# Leaderboard, the CDC's way: pairwise relative WIS, on counts and on log counts", "",
             "Pairwise relative WIS (Cramer et al. 2022) vs the CDC FluSight-baseline, below 1 = better.",
             "Log scale = WIS of log(x+1), the scale the hub says final rankings use. Ranked if a model",
             "covered >= 75% of Sentinel's forecasts that season (50 states, horizons 0-3). Sentinel = replay with data as published",
             "each week, quantiles rounded to whole numbers; v2 and v3 enter separately.",
             "**Honesty labels:** v2 was designed on 2023-24 and 2024-25 and is blind on 2025-26; v3 was",
             "chosen after seeing all three seasons (in-sample everywhere). Other teams forecast live.", ""]
    summary = []
    for season in ("2023-24", "2024-25", "2025-26"):
        sm = (keys["season"] == season).values
        n = sm.sum()
        for scale in ("log", "count"):
            f = (lambda a: np.log1p(np.maximum(a, 0))) if scale == "log" else (lambda a: a)
            cols = {}
            for name, Q in hub_q.items():
                Qm = Q[sm]
                have = ~np.isnan(Qm).any(axis=1)
                if have.sum() < 0.75 * n and name != "FluSight-baseline":
                    continue
                wv = np.full(n, np.nan)
                wv[have] = wis_rows(f(Qm[have]), f(y[sm][have]))
                cols[name] = wv
            if "FluSight-baseline" not in cols:
                continue
            ranks = {}
            for ours_name, Q in Qs.items():
                c = dict(cols)
                c[ours_name] = wis_rows(f(Q[sm]), f(y[sm]))
                names = list(c)
                rel = pairwise(np.column_stack([c[k] for k in names]), names, "FluSight-baseline")
                rel = rel.drop("FluSight-baseline").sort_values()
                ranks[ours_name] = (int(np.where(rel.index == ours_name)[0][0]) + 1, len(rel), rel[ours_name], rel)
            r2, r3 = ranks["Sentinel-v2 (replay)"], ranks["Sentinel-v3 (replay)"]
            summary.append({"season": season, "scale": scale,
                            "v2 rank": f"{r2[0]} of {r2[1]}", "v2": round(r2[2], 3),
                            "v3 rank": f"{r3[0]} of {r3[1]}", "v3": round(r3[2], 3),
                            "CDC ensemble": round(r2[3].get("FluSight-ensemble", np.nan), 3),
                            "best other team": f"{r2[3].drop(r2[3].index[r2[3].index.str.startswith('Sentinel')]).index[0]} "
                                               f"{r2[3].drop(r2[3].index[r2[3].index.str.startswith('Sentinel')]).iloc[0]:.3f}"})
            lines.append(f"## {season}, {scale} scale: v2 ranks {r2[0]} of {r2[1]} ({r2[2]:.3f}), "
                         f"v3 ranks {r3[0]} of {r3[1]} ({r3[2]:.3f})")
            top = r3[3].head(12)
            tbl = pd.DataFrame({"model": top.index, "relative WIS": top.values.round(3)})
            tbl.insert(0, "rank", range(1, len(tbl) + 1))
            lines += ["", "Top 12 (tournament with v3):", "", tbl.to_markdown(index=False), ""]
            print(f"{season} {scale}: v2 {r2[0]}/{r2[1]} {r2[2]:.3f}  v3 {r3[0]}/{r3[1]} {r3[2]:.3f}", flush=True)
    head = lines[:8] + ["## Summary", "", "(CDC ensemble and best other team: from the tournament with v2.)", "",
                        pd.DataFrame(summary).to_markdown(index=False), ""]
    out.write_text("\n".join(head + lines[8:]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hub", type=Path, default=ROOT.parent / "FluSight-forecast-hub")
    ap.add_argument("--out", type=Path, default=ROOT / "backtest/leaderboard_official.md")
    a = ap.parse_args()
    main(a.hub, a.out)
