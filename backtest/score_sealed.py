"""Score the sealed forecasts (submission/sealed/) once the CDC has published the weeks they forecast.

    python -m backtest.score_sealed [--truth ../FluSight-forecast-hub/target-data/target-hospital-admissions.csv]

Prints, for every target week already published: the whole-US numbers (v2, v3, actual) and the
WIS of v2, v3 and a flat baseline over the 50 states (lower = better), on counts and on log counts.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.wis import QUANTILES

ROOT = Path(__file__).resolve().parents[1]
FILES = {"v3": ROOT / "submission/sealed/2026-10-03-Sentinel-Ens.csv",
         "v2": ROOT / "submission/sealed/v2/2026-10-03-Sentinel-Ens.csv"}


def wis(Q: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 2.0 * ((((y[:, None] <= Q).astype(float)) - QUANTILES[None, :]) * (Q - y[:, None])).mean(axis=1)


def main(truth_path: Path):
    t = pd.read_csv(truth_path, dtype={"location": str}, parse_dates=["date"])
    truth = t.set_index(["location", "date"])["value"]
    rows = []
    for name, f in FILES.items():
        d = pd.read_csv(f, dtype={"location": str, "output_type_id": str}, parse_dates=["target_end_date"])
        w = d.pivot_table(index=["location", "target_end_date"], columns="output_type_id", values="value")
        w = w[[f"{q:g}" for q in QUANTILES]]
        for (loc, date), q in w.iterrows():
            y = truth.get((loc, date))
            if y is not None and not np.isnan(y):
                rows.append((name, loc, date, y, q.values.astype(float)))
    if not rows:
        print("None of the forecast weeks has been published yet.")
        return
    # Sentinel's flat baseline from the same data the sealed forecasts had (published 2026-09-30)
    from models.models import baseline
    ts = pd.read_csv(ROOT.parent / "FluSight-forecast-hub/target-data/time-series.csv", dtype={"location": str},
                     parse_dates=["as_of", "target_end_date"])
    v = ts[(ts["target"] == "wk inc flu hosp") & (ts["as_of"] == "2026-09-30")]
    last = pd.Timestamp("2026-09-26")
    for loc, g in v.groupby("location"):
        series = g.set_index("target_end_date")["observation"].sort_index().dropna()
        for k in (1, 2, 3, 4):
            date = last + pd.Timedelta(weeks=k)
            y = truth.get((loc, date))
            if y is not None and not np.isnan(y) and len(series) >= 30:
                rows.append(("baseline", loc, date, y, np.asarray(baseline(series, k), float)))
    r = pd.DataFrame(rows, columns=["model", "location", "week", "actual", "q"])
    Q, y = np.vstack(r["q"].values), r["actual"].values.astype(float)
    r["wis"] = wis(Q, y)
    r["wis_log"] = wis(np.log1p(Q), np.log1p(y))
    r["median"] = Q[:, 11]
    r["in95"] = (Q[:, 1] <= y) & (y <= Q[:, 21])
    us = r[r["location"] == "US"].pivot_table(index="week", columns="model", values="median")
    us["actual"] = r[r["location"] == "US"].groupby("week")["actual"].first()
    print("Whole US, patients (medians vs actual):")
    print(us.round(0).astype(int).to_string())
    st = r[~r["location"].isin(["US", "11", "72"])]
    print("\n50 states, total WIS by week (lower = better):")
    print(st.pivot_table(index="week", columns="model", values=["wis", "wis_log"], aggfunc="sum").round(1).to_string())
    print("\n95% range held the truth:", st.groupby("model")["in95"].mean().round(2).to_dict())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", type=Path,
                    default=ROOT.parent / "FluSight-forecast-hub/target-data/target-hospital-admissions.csv")
    main(ap.parse_args().truth)
