"""Official replay of Sentinel v2 (models/engine.py) over past seasons, week by week, with the hospital
data exactly as published each week. Scores against final data, like the CDC.

Writes backtest/results_v2.csv with three models side by side on identical forecasts:
  baseline   CDC-style flat baseline
  v1         the previous Sentinel design (for an honest before/after)
  ensemble   Sentinel v2 (the engine that the live weekly forecast uses)

Run:  python -m backtest.run_v2 [--seasons 2023-24 2024-25 2025-26]
Then: python -m backtest.compare_cdc --results backtest/results_v2.csv --out backtest/vs_cdc_v2.md
      python -m backtest.leaderboard --results backtest/results_v2.csv --out backtest/leaderboard_v2.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.harness import DEV, HOLDOUT, Replay
from models import engine

ROOT = Path(__file__).resolve().parents[1]
ED_HAIRCUT = 1.028  # newest ER week's typical later upward revision (measured from CDC's archive)


class EdArchive:
    """ER-visit data as it was published on a given day: CDC's archived version where one exists
    (Dec 2025 on), otherwise the final numbers with the newest week scaled down by its typical
    later revision, so it behaves like a first publication."""

    def __init__(self, hub: Path, final_ed: pd.DataFrame):
        t = pd.read_csv(hub / "target-data/time-series.csv", dtype={"location": str},
                        parse_dates=["as_of", "target_end_date"])
        t = t[t["target"] == "wk inc flu prop ed visits"]
        self.t = t.rename(columns={"target_end_date": "date", "observation": "value"})
        self.final = final_ed

    def as_published(self, as_of: pd.Timestamp, last: pd.Timestamp) -> pd.DataFrame:
        """With an archived version (since the 2025-26 season, when CDC's hub publishes ER visits the
        same day as hospital data): exactly that, newest week = `last`. Without one (earlier, when the
        NSSP data came out on Fridays, two days AFTER the Wednesday deadline): only up to `last` - 1
        week, the newest of those scaled down to look first-published. The engine then automatically
        uses ER visits one week behind."""
        vin = self.t[self.t["as_of"] == as_of]
        if len(vin):
            v = vin[["date", "location", "value"]].copy()
            v["ed_level"] = np.log(v["value"] + 1e-4)
            return v[["date", "location", "ed_level"]]
        newest_avail = last - pd.Timedelta(weeks=1)
        e = self.final[self.final["date"] <= newest_avail].copy()
        newest = e["date"] == newest_avail
        e.loc[newest, "ed_level"] = e.loc[newest, "ed_level"] - np.log(ED_HAIRCUT)
        return e

    def as_published_today(self, as_of: pd.Timestamp, last: pd.Timestamp) -> pd.DataFrame:
        """Round 32: like as_published, but simulating today's schedule (ER visits out the same Wednesday as
        hospital data) when no archived version exists: final numbers up to `last`, the newest week scaled
        down by its typical later revision. With an archived version: exactly that version."""
        vin = self.t[self.t["as_of"] == as_of]
        if len(vin):
            return self.as_published(as_of, last)
        e = self.final[self.final["date"] <= last].copy()
        newest = e["date"] == last
        e.loc[newest, "ed_level"] = e.loc[newest, "ed_level"] - np.log(ED_HAIRCUT)
        return e


def make_v2_forecaster(rep: Replay, cfg: engine.Config = engine.V2, hub: Path | None = None):
    hub = hub or ROOT.parent / "FluSight-forecast-hub"
    ed = EdArchive(hub, rep.sig["ed"])
    lags = engine.lag_table(rep.vint.t)
    pop = pd.read_csv(ROOT / "data/clean/locations.csv", dtype={"location": str}).set_index("location")["population"]
    hist = engine.load_history(cfg) if cfg.hist else None

    def f(w):
        hosp = w.v[~w.v["location"].isin(["US", "11", "72"])]
        sig = {"ww": rep.sig["ww"], "weather": rep.sig["weather"], "ed": ed.as_published(w.as_of, w.last)}
        return engine.forecast(hosp, w.last, w.as_of, sig, lags, pop, cfg, states_only=True, hist=hist)
    return f


def main(seasons: tuple, out: Path):
    from backtest.experiments import CURRENT, make_forecaster
    rep = Replay(seasons)
    print(f"{len(rep.weeks)} weeks", flush=True)
    b = rep.score(rep.baseline_forecaster, "baseline")
    v1 = rep.score(make_forecaster(rep, CURRENT), "v1")
    v2 = rep.score(make_v2_forecaster(rep), "ensemble")
    key = ["season", "reference_date", "location", "horizon"]
    common = b[key].merge(v1[key], on=key).merge(v2[key], on=key)
    res = pd.concat([d.merge(common, on=key) for d in (b, v1, v2)])
    res["reference_date"] = pd.to_datetime(res["reference_date"]).dt.date
    res.to_csv(out, index=False)
    report(res, out)


def report(res: pd.DataFrame, out: Path):
    for s, g in res.groupby("season"):
        m = g.groupby("model")["wis"].mean()
        print(f"{s}: v1 {m['v1'] / m['baseline']:.3f}  v2 {m['ensemble'] / m['baseline']:.3f}")
    m = res.groupby("model")["wis"].mean()
    print(f"ALL: v1 {m['v1'] / m['baseline']:.3f}  v2 {m['ensemble'] / m['baseline']:.3f}  -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", default=list(DEV + HOLDOUT))
    ap.add_argument("--out", type=Path, default=ROOT / "backtest/results_v2.csv")
    ap.add_argument("--merge", nargs="+", type=Path, default=None,
                    help="merge per-season result files into --out instead of running")
    a = ap.parse_args()
    if a.merge:
        res = pd.concat([pd.read_csv(f, dtype={"location": str}) for f in a.merge])
        res.to_csv(a.out, index=False)
        report(res, a.out)
    else:
        main(tuple(a.seasons), a.out)
