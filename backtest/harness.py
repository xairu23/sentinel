"""Fast experiment harness.

Replays past seasons week by week using the hospital data EXACTLY as published that week (the CDC
hub's archived versions), and scores any candidate forecaster against final data with WIS.

Discipline (see HOW_IT_WORKS.md, "Honesty check"):
  DEV seasons (2023-24, 2024-25) are the ONLY ones used to decide anything.
  HOLDOUT (2025-26) is scored once, at the end, after the design is locked.

A forecaster is any function  f(week) -> {(location, k): array of 23 quantiles}.
`week` carries everything that was knowable on that Wednesday: the published hospital data,
the signals (with their delays), the features, and the archive of earlier published versions.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.run_backtest import SEASONS, Vintages, attach_signals, final_hosp, load_signals
from backtest.wis import QUANTILES
from models.models import K_STEPS, baseline, load_neighbor_pairs, make_features

ROOT = Path(__file__).resolve().parents[1]
DEV = ("2023-24", "2024-25")
HOLDOUT = ("2025-26",)
EXCLUDE = {"US", "11", "72"}  # scored on the 50 states, same as the official backtest


@dataclass
class Week:
    season: str
    ref: pd.Timestamp
    last: pd.Timestamp
    as_of: pd.Timestamp
    panel: pd.DataFrame          # published hospital data + delayed signals (50 states)
    feat: pd.DataFrame           # features built from `panel`
    v: pd.DataFrame = None       # the raw published hospital data for that week
    cache: dict = field(default_factory=dict)


class Replay:
    def __init__(self, seasons=DEV, hub: Path | None = None):
        load_neighbor_pairs(ROOT / "data/clean/locations.csv")
        self.sig = load_signals()
        self.vint = Vintages(hub or ROOT.parent / "FluSight-forecast-hub")
        final = attach_signals(final_hosp(), self.sig)
        self.truth = final.set_index(["location", "date"])["value"]
        self.weeks: list[Week] = []
        for season in seasons:
            start, end = SEASONS[season]
            for ref in pd.date_range(start, end, freq="7D"):
                last = ref - pd.Timedelta(weeks=1)
                ok = self.vint.max_date[self.vint.max_date == last]
                if ok.empty:
                    continue
                as_of = ok.index.min()
                v = self.vint.t[self.vint.t["as_of"] == as_of][["date", "location", "value"]]
                panel = attach_signals(v, self.sig)
                self.weeks.append(Week(season, ref, last, as_of, panel, make_features(panel), v=v))

    # ---- earlier published versions, strictly before this week's ----------------------------
    def history_before(self, week: Week) -> pd.DataFrame:
        t = self.vint.t
        return t[t["as_of"] < week.as_of]

    # ---- scoring --------------------------------------------------------------------------------
    def baseline_forecaster(self, week: Week) -> dict:
        out = {}
        for loc, s in week.panel[week.panel["date"] <= week.last].groupby("location"):
            s = s.set_index("date")["value"].dropna()
            if len(s) < 30:
                continue
            for k in K_STEPS:
                out[(loc, k)] = baseline(s, k)
        return out

    def score(self, forecaster, name: str = "model") -> pd.DataFrame:
        rows = []
        for w in self.weeks:
            preds = forecaster(w)
            for (loc, k), q in preds.items():
                if loc in EXCLUDE or q is None:
                    continue
                y = self.truth.get((loc, w.last + pd.Timedelta(weeks=k)))
                if y is None or np.isnan(y):
                    continue
                rows.append((w.season, w.ref, loc, k - 1, y, np.asarray(q, float)))
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=["season", "reference_date", "location", "horizon", "observed", "q"])
        Q = np.vstack(df["q"].values)
        y = df["observed"].values[:, None]
        pin = ((y <= Q).astype(float) - QUANTILES) * (Q - y)
        df["wis"] = 2.0 * pin.mean(axis=1)
        lo50, hi50 = Q[:, QUANTILES == 0.25][:, 0], Q[:, QUANTILES == 0.75][:, 0]
        lo95, hi95 = Q[:, QUANTILES == 0.025][:, 0], Q[:, QUANTILES == 0.975][:, 0]
        df["cov50"] = (lo50 <= df["observed"]) & (df["observed"] <= hi50)
        df["cov95"] = (lo95 <= df["observed"]) & (df["observed"] <= hi95)
        df["median"] = Q[:, QUANTILES == 0.5][:, 0]
        df["model"] = name
        return df.drop(columns="q")


def paired(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    key = ["season", "reference_date", "location", "horizon"]
    return a[key + ["wis"]].merge(b[key + ["wis"]], on=key, suffixes=("_a", "_b"))


def compare(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """% change in mean WIS of b vs a on their shared forecasts, 95% CI resampling whole weeks."""
    p = paired(a, b)
    by_week = p.groupby("reference_date")[["wis_a", "wis_b"]].sum()
    weeks = by_week.index.values
    rng = np.random.default_rng(seed)
    point = (p["wis_b"].sum() / p["wis_a"].sum() - 1) * 100
    idx = rng.integers(0, len(weeks), size=(n_boot, len(weeks)))
    A, B = by_week["wis_a"].values, by_week["wis_b"].values
    boots = (B[idx].sum(1) / A[idx].sum(1) - 1) * 100
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi


def rel(a: pd.DataFrame, base: pd.DataFrame) -> float:
    p = paired(base, a)
    return p["wis_b"].sum() / p["wis_a"].sum()


def verdict(lo: float, hi: float) -> str:
    return "HELPS" if hi < 0 else ("HURTS" if lo > 0 else "unclear")
