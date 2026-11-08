"""The live scoreboard: every forecast Sentinel has made this season, scored against what really
happened, as soon as the CDC publishes it.

Scores the submitted files (submission/output), the v2 shadow forecasts (submission/shadow/v2, made the
same night but never submitted) and the sealed forecasts (submission/sealed), plus the CDC's own
FluSight-baseline and FluSight-ensemble when their files for the same round are in the hub copy.
Truth = the latest published numbers (they can still be revised a little).

    python -m backtest.score_live
Writes backtest/live_scoreboard.md
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backtest.wis import QUANTILES

ROOT = Path(__file__).resolve().parents[1]
HUB = ROOT.parent / "FluSight-forecast-hub"
QSTR = [f"{q:g}" for q in QUANTILES]
SOURCES = {"v3 (submitted)": ROOT / "submission/output", "v2 (shadow)": ROOT / "submission/shadow/v2",
           "v3 (sealed)": ROOT / "submission/sealed", "v2 (sealed)": ROOT / "submission/sealed/v2"}


def wis(Q: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 2.0 * ((((y[:, None] <= Q).astype(float)) - QUANTILES[None, :]) * (Q - y[:, None])).mean(axis=1)


def load(f: Path, model: str) -> pd.DataFrame:
    d = pd.read_csv(f, dtype={"location": str, "output_type_id": str})
    d = d[(d["target"] == "wk inc flu hosp") & (d["output_type"] == "quantile") & (d["horizon"] >= 0)]
    d["output_type_id"] = d["output_type_id"].astype(float).map(lambda x: f"{x:g}")
    w = d.pivot_table(index=["reference_date", "location", "horizon", "target_end_date"],
                      columns="output_type_id", values="value").reset_index()
    w["model"] = model
    return w


def main():
    truth = pd.read_csv(HUB / "target-data/target-hospital-admissions.csv", dtype={"location": str})
    truth = truth.set_index(["location", "date"])["value"]
    parts, rounds = [], set()
    for name, folder in SOURCES.items():
        for f in sorted(folder.glob("*-Sentinel-Ens.csv")):
            parts.append(load(f, name))
            rounds.add(f.name[:10])
    for m in ("FluSight-baseline", "FluSight-ensemble"):
        for r in sorted(rounds):
            f = HUB / "model-output" / m / f"{r}-{m}.csv"
            if f.exists():
                parts.append(load(f, m))
    if not parts:
        print("no forecasts yet")
        return
    d = pd.concat(parts, ignore_index=True)
    d["y"] = [truth.get((l, t), np.nan) for l, t in zip(d["location"], d["target_end_date"])]
    d = d[d["y"].notna()].copy()
    if d.empty:
        print("None of the forecast weeks has been published yet.")
        return
    Q, y = d[QSTR].values.astype(float), d["y"].values.astype(float)
    d["wis"], d["wis_log"] = wis(Q, y), wis(np.log1p(Q), np.log1p(y))
    d["in95"] = (Q[:, 1] <= y) & (y <= Q[:, 21])
    lines = ["# Live scoreboard, 2026-27 season", "",
             "WIS on log(x+1) (the CDC's ranking scale) and on counts, all places each model forecast;",
             "lower is better. Only weeks the CDC has already published.", ""]
    for r, g in d.groupby("reference_date"):
        t = g.groupby("model").agg(forecasts=("wis", "size"), log_wis=("wis_log", "mean"), wis=("wis", "mean"),
                                   in95=("in95", "mean")).round(3).sort_values("log_wis")
        lines += [f"## Round {r}", "", t.to_markdown(), ""]
        us = g[g["location"] == "US"].pivot_table(index="target_end_date", columns="model", values="0.5")
        if len(us):
            us["actual"] = g[g["location"] == "US"].groupby("target_end_date")["y"].first()
            lines += ["Whole US, median forecast vs actual:", "", us.round(0).to_markdown(), ""]
    out = "\n".join(lines)
    (ROOT / "backtest/live_scoreboard.md").write_text(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
