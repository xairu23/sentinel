"""Where would Sentinel have ranked among ALL teams in past FluSight seasons?

For each season, every hub model is scored (WIS, against final data) on the forecasts it shares with
Sentinel's real-time backtest. Each model's score is its mean WIS divided by the CDC FluSight-baseline's
mean WIS on exactly the same forecasts (relative WIS, below 1 = better). A model is ranked in a season only
if it covered at least 75% of Sentinel's forecasts that season (the hub's own inclusion rule uses 75%).

Run from the repo root:  python -m backtest.leaderboard --hub ../FluSight-forecast-hub
Output: backtest/leaderboard.md

Caveats (written into the output): Sentinel's numbers come from a replay with as-published hospital data,
not from live submissions, and its model was designed with these seasons in view. Other teams forecast live.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.wis import QUANTILES

ROOT = Path(__file__).resolve().parents[1]
QSTR = [f"{q:g}" for q in QUANTILES]


def vector_wis(qmat: np.ndarray, y: np.ndarray) -> np.ndarray:
    """WIS for many forecasts at once. qmat: (n, 23) quantile values, y: (n,) observations."""
    ind = (y[:, None] <= qmat).astype(float)
    return 2.0 * (((ind - QUANTILES[None, :]) * (qmat - y[:, None])).mean(axis=1))


def load_model(folder: Path, keys: pd.DataFrame) -> pd.DataFrame | None:
    parts = []
    for f in sorted(folder.glob("*.csv")):
        ref = f.name[:10]
        if ref not in keys["ref"].values:
            continue
        try:
            d = pd.read_csv(f, dtype={"location": str, "output_type_id": str},
                            usecols=["reference_date", "target", "horizon", "location", "output_type",
                                     "output_type_id", "value"])
        except (ValueError, pd.errors.ParserError):
            continue
        d = d[(d["target"] == "wk inc flu hosp") & (d["output_type"] == "quantile") & (d["horizon"].between(0, 3))]
        parts.append(d)
    if not parts:
        return None
    d = pd.concat(parts)
    d["output_type_id"] = d["output_type_id"].astype(float).map(lambda x: f"{x:g}")
    d = d[d["output_type_id"].isin(QSTR)]
    w = d.pivot_table(index=["reference_date", "location", "horizon"], columns="output_type_id",
                      values="value", aggfunc="first")
    if not set(QSTR).issubset(w.columns):
        return None
    w = w[QSTR].dropna()
    return w


def main(hub: Path, results: Path | None = None, out: Path | None = None):
    ours = pd.read_csv(results or ROOT / "backtest/results_scores_realtime.csv", dtype={"location": str})
    ours = ours[ours["model"] == "ensemble"].rename(columns={"reference_date": "ref"})
    ours["ref"] = ours["ref"].astype(str)
    keys = ours[["season", "ref", "location", "horizon", "observed", "wis"]]
    obs = keys.set_index(["ref", "location", "horizon"])["observed"]

    scores = {"Sentinel-Ens (replay)": keys.set_index(["ref", "location", "horizon"])["wis"]}
    folders = [p for p in (hub / "model-output").iterdir() if p.is_dir()]
    for i, folder in enumerate(sorted(folders)):
        w = load_model(folder, keys)
        if w is None or w.empty:
            continue
        w.index = w.index.set_names(["ref", "location", "horizon"])
        w = w[w.index.isin(obs.index)]
        if w.empty:
            continue
        y = obs.loc[w.index].values
        scores[folder.name] = pd.Series(vector_wis(w.values.astype(float), y), index=w.index)
        print(f"{i + 1}/{len(folders)} {folder.name}: {len(w):,} forecasts", flush=True)

    allw = pd.DataFrame(scores)
    season_of = keys.set_index(["ref", "location", "horizon"])["season"]
    allw["season"] = season_of.loc[allw.index].values
    lines = ["# Leaderboard: Sentinel replay vs every team in the CDC FluSight hub", "",
             "Relative WIS vs the CDC FluSight-baseline on the forecasts each model shares with Sentinel",
             "(below 1 = better than baseline). Ranked only if the model covered >= 75% of Sentinel's",
             "forecasts that season.", ""]
    for season, g in allw.groupby("season"):
        n_keys = len(g)
        rows = []
        for m in g.columns.drop("season"):
            if m == "FluSight-baseline":
                continue
            both = g[[m, "FluSight-baseline"]].dropna()
            if len(both) < 0.75 * n_keys:
                continue
            rows.append((m, both[m].mean() / both["FluSight-baseline"].mean(), len(both)))
        lb = pd.DataFrame(rows, columns=["model", "relative WIS", "forecasts"]).sort_values("relative WIS")
        lb.insert(0, "rank", range(1, len(lb) + 1))
        r = lb.loc[lb["model"] == "Sentinel-Ens (replay)", "rank"]
        own = lb.loc[lb["model"] == "Sentinel-Ens (replay)", "relative WIS"]
        lines += [f"## Season {season}: Sentinel replay ranks {int(r.iloc[0]) if len(r) else 'n/a'} of {len(lb)}"
                  + (f" (relative WIS {own.iloc[0]:.3f})" if len(own) else ""), "",
                  lb.head(15).round(3).to_markdown(index=False), ""]
        print(f"{season}: Sentinel rank {int(r.iloc[0]) if len(r) else 'n/a'} of {len(lb)}")
    lines += ["## Caveats", "",
              "- Sentinel's forecasts are a replay using hospital data as published each week, not live submissions.",
              "- Sentinel's model was designed with these seasons in view; other teams forecast live, unseen.",
              "- Teams cover different weeks and locations; each is compared to the baseline on its shared forecasts.",
              "- The official CDC evaluation uses a pairwise method; this is a simpler, transparent version."]
    (out or ROOT / "backtest/leaderboard.md").write_text("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hub", type=Path, default=ROOT.parent / "FluSight-forecast-hub")
    ap.add_argument("--results", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    main(a.hub, a.results, a.out)
