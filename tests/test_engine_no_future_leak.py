"""Leak tests for Sentinel v2 (models/engine.py).   Run:  python -m tests.test_engine_no_future_leak

1. Future check: a forecast for week `last` must be IDENTICAL whether or not anything published
   after it exists -- later hospital numbers, later signals, later versions in CDC's archive.
   The hospital data used is CDC's archived version as published that week (not final data).
2. Delay check: wastewater is used one week behind, so deleting the wastewater value for week
   `last` itself must change NOTHING. (This is the check that catches a delay being undone by
   gap-filling, the bug found by the independent review.)
3. ER-availability check: when the newest ER week is older than `last` (as before the 2025-26
   season), the engine must behave exactly as with ER one week behind.
Run with and without the long history, for the 50 states and for all 53 places.
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.run_backtest import Vintages, load_signals
from models import engine

ROOT = Path(__file__).resolve().parents[1]
HUB = ROOT.parent / "FluSight-forecast-hub"
VINT = None


def _same(a: dict, b: dict) -> int:
    keys = set(a) | set(b)
    return sum(1 for k in keys if k not in a or k not in b or not np.allclose(a[k], b[k]))


def setup(last: str, scope: str):
    global VINT
    VINT = VINT or Vintages(HUB)
    last = pd.Timestamp(last)
    as_of = VINT.max_date[VINT.max_date == last].index.min()
    hosp = VINT.t[VINT.t["as_of"] == as_of][["date", "location", "value"]]  # as published that week
    if scope == "states":
        hosp = hosp[~hosp["location"].isin(["US", "11", "72"])]
    pop = pd.read_csv(ROOT / "data/clean/locations.csv", dtype={"location": str}).set_index("location")["population"]
    sig = {k: v for k, v in load_signals().items() if k in ("ww", "ed", "weather")}
    return last, as_of, hosp, pop, sig


def check(last: str, cfg: engine.Config, scope: str) -> int:
    last, as_of, hosp, pop, sig = setup(last, scope)
    so = scope == "states"
    hist = engine.load_history(cfg) if cfg.hist else None
    run = lambda h, s, lags: engine.forecast(h, last, as_of, s, lags, pop, cfg, states_only=so, hist=hist)
    full = run(hosp, sig, engine.lag_table(VINT.t))
    # 1. delete everything published after `last` / after this version
    cut_sig = {k: v[v["date"] <= last] for k, v in sig.items()}
    cut = run(hosp[hosp["date"] <= last], cut_sig, engine.lag_table(VINT.t[VINT.t["as_of"] <= as_of]))
    bad1 = _same(full, cut)
    # 2. wastewater for week `last` itself must not be used (one-week delay)
    no_ww_last = dict(cut_sig, ww=cut_sig["ww"][cut_sig["ww"]["date"] < last])
    bad2 = _same(cut, run(hosp, no_ww_last, engine.lag_table(VINT.t[VINT.t["as_of"] <= as_of])))
    print(f"{last.date()} hist={cfg.hist} gbq={cfg.gbq_weight} families={cfg.families} scope={scope}: {len(full):,} forecasts; "
          f"future check differences {bad1}; wastewater-delay check differences {bad2}", flush=True)
    return bad1 + bad2


def check_ed_availability(last: str) -> int:
    """ER newest week older than `last` -> must equal an explicit one-week ER delay."""
    last, as_of, hosp, pop, sig = setup(last, "states")
    sig = dict(sig, ed=sig["ed"][sig["ed"]["date"] < last])
    lags = engine.lag_table(VINT.t[VINT.t["as_of"] <= as_of])
    auto = engine.forecast(hosp, last, as_of, sig, lags, pop, engine.LINEAR, states_only=True)
    explicit = engine.forecast(hosp, last, as_of, sig, lags, pop, replace(engine.LINEAR, ed_delay=1),
                               states_only=True)
    bad = _same(auto, explicit)
    print(f"{last.date()} ER-availability check: {len(auto):,} forecasts, differences {bad}", flush=True)
    return bad


if __name__ == "__main__":
    import os
    os.environ.setdefault("GBQ_THREADS", "1")  # single-threaded trees: bit-for-bit repeatable
    total = 0
    for scope in ("states", "all"):  # the submitted design (v3: regressions + trees + quantile autoregression)
        for d in ("2024-01-06", "2025-01-25", "2026-02-14"):
            total += check(d, engine.FINAL, scope)
    for d in ("2024-01-06", "2025-01-25"):  # the long-history option (Experiment 9, not used)
        total += check(d, replace(engine.LINEAR, hist=True), "states")
    for d in ("2024-01-06", "2025-01-25"):
        total += check_ed_availability(d)
    print("PASS: the engine uses no future data" if total == 0 else f"FAIL: {total} forecasts changed")
