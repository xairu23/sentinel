"""Benchmark Sentinel's backtest against the CDC's real published forecasts from past seasons.

Reads every past forecast of FluSight-baseline and FluSight-ensemble from a local copy of the CDC hub,
scores them with the same WIS and the same observed values, and compares on exactly the same
(reference date, location, horizon) as Sentinel's backtest.

Run from the repo root:  python -m backtest.compare_cdc --mode realtime --hub ../FluSight-forecast-hub
Output: backtest/vs_cdc_realtime.md (or vs_cdc.md with --mode revised)

IMPORTANT fairness note (written into the output): the CDC models forecast in real time with
preliminary data; Sentinel's backtest uses today's corrected data. That gives Sentinel an advantage,
so this table is an optimistic upper bound for Sentinel, not a like-for-like win.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.wis import QUANTILES, wis

ROOT = Path(__file__).resolve().parents[1]
CDC_MODELS = ["FluSight-baseline", "FluSight-ensemble"]


def load_cdc(hub: Path, model: str) -> pd.DataFrame:
    parts = []
    for f in sorted((hub / "model-output" / model).glob("*.csv")):
        d = pd.read_csv(f, dtype={"location": str, "output_type_id": str})
        d = d[(d["target"] == "wk inc flu hosp") & (d["output_type"] == "quantile") & (d["horizon"] >= 0)]
        parts.append(d)
    d = pd.concat(parts)
    d["q"] = d["output_type_id"].astype(float).round(3)
    d = d[d["q"].isin(np.round(QUANTILES, 3))]
    return d


def main(hub: Path, mode: str, results: Path | None = None, out: Path | None = None):
    suffix = "" if mode == "revised" else "_realtime"
    ours = pd.read_csv(results or ROOT / f"backtest/results_scores{suffix}.csv", dtype={"location": str})
    ours["reference_date"] = pd.to_datetime(ours["reference_date"])
    truth = pd.read_csv(ROOT / "data/clean/hosp_flu.csv", dtype={"location": str}, parse_dates=["date"])
    truth = truth.set_index(["location", "date"])["value"]

    keys = ours[ours["model"] == "baseline"][["season", "reference_date", "location", "horizon"]]
    rows = []
    for model in CDC_MODELS:
        d = load_cdc(hub, model)
        d["reference_date"] = pd.to_datetime(d["reference_date"])
        d["target_end_date"] = pd.to_datetime(d["target_end_date"])
        for (ref, loc, h, ted), g in d.groupby(["reference_date", "location", "horizon", "target_end_date"]):
            g = g.sort_values("q")
            if len(g) != len(QUANTILES):
                continue
            y = truth.get((loc, ted))
            if y is None or np.isnan(y):
                continue
            rows.append({"reference_date": ref, "location": loc, "horizon": h, "model": model,
                         "wis": wis(g["value"].values, y)})
    cdc = pd.DataFrame(rows).merge(keys, on=["reference_date", "location", "horizon"], how="inner")
    both = pd.concat([ours[["season", "reference_date", "location", "horizon", "model", "wis"]]
                      .assign(model=lambda x: "Sentinel-" + x["model"]), cdc])
    piv = both.pivot_table(index=["season", "reference_date", "location", "horizon"],
                           columns="model", values="wis").dropna()  # only keys every model has
    mean = piv.mean()
    rel = (mean / mean["FluSight-baseline"]).sort_values()
    by_season = piv.groupby("season").mean()
    by_season = by_season.div(by_season["FluSight-baseline"], axis=0).round(3)

    lines = [f"# Sentinel vs the CDC's real forecasts (past seasons) - Sentinel using {'revised' if mode == 'revised' else 'real-time (as published)'} data", "",
             f"Compared on {len(piv):,} identical (week, location, horizon) forecasts.",
             "Relative WIS vs the CDC FluSight-baseline (below 1 = better).", "",
             rel.round(3).rename("relative WIS").to_frame().to_markdown(), "",
             "## By season", "", by_season.T.to_markdown(), "",
             "## Fairness note", ""]
    if mode == "revised":
        lines += ["The CDC models forecast in real time with preliminary, later-corrected data. This Sentinel run",
                  "used today's corrected data, which makes the task easier: an optimistic upper bound."]
    else:
        lines += ["Sentinel used the hospital data exactly as first published each week (the same information the",
                  "CDC teams had), so this is a like-for-like comparison for hospital data. Remaining differences:",
                  "wastewater has no archived versions (used with a 1-week delay), and Sentinel's features were",
                  "chosen with these seasons in view. The live 2026-27 season is the final, fully fair test."]
    (out or ROOT / f"backtest/vs_cdc{suffix}.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hub", type=Path, default=ROOT.parent / "FluSight-forecast-hub")
    ap.add_argument("--mode", choices=["revised", "realtime"], default="realtime")
    ap.add_argument("--results", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    main(a.hub, a.mode, a.results, a.out)
