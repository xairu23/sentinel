"""Leak test: a forecast made on week `last` must be IDENTICAL whether or not the data after `last`
exists. If any number changes, some clue is secretly using the future.  Run:  python -m tests.test_no_future_leak"""
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.run_backtest import attach_signals, final_hosp, load_signals
from models.models import MODELS, load_neighbor_pairs, make_features, run_all

ROOT = Path(__file__).resolve().parents[1]


def check(last: str) -> int:
    last = pd.Timestamp(last)
    sig = load_signals()
    full = attach_signals(final_hosp(), sig)
    cut_h = final_hosp()[lambda d: d["date"] <= last]
    cut_sig = {k: v[v["date"] <= last] for k, v in sig.items()}
    cut = attach_signals(cut_h, cut_sig)
    a, b = run_all(make_features(full), last), run_all(make_features(cut), last)
    all_models = set(a) | set(b)  # MODELS plus anything run_all adds on top (e.g. the GBM model)
    bad = 0
    for m in all_models:
        keys = set(a.get(m, {})) | set(b.get(m, {}))
        for key in keys:
            if key not in a.get(m, {}) or key not in b.get(m, {}) or not np.allclose(a[m][key], b[m][key]):
                bad += 1
    print(f"{last.date()}: {sum(len(a[m]) for m in all_models):,} forecasts compared, differences: {bad}")
    return bad


if __name__ == "__main__":
    load_neighbor_pairs(ROOT / "data/clean/locations.csv")
    total = sum(check(d) for d in ["2024-01-06", "2025-01-25", "2026-02-14"])
    print("PASS: no model uses future data" if total == 0 else f"FAIL: {total} forecasts changed")
