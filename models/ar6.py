"""Sentinel AR6: pooled autoregression of order 6 on fourth-root rates (round-3 candidate, idea 11 in
backtest/V3_PROTOCOL.md).

After the AR(6) component UMass described for flusion in 2024-25 and 2025-26 (hub file
model-metadata/UMass-flusion.yml: "AR(6) pooled model with shared coefficients across locations and
per-location variance", fit to NHSN hospital data only):
  * y = (admissions per 100k + 0.01) ** (1/4), one weekly series per place,
  * y_t = a + phi_1 y_(t-1) + ... + phi_6 y_(t-6) + noise, ONE set of coefficients for all places
    (ordinary least squares on the last 3 years),
  * each place's own noise level (standard deviation of its own residuals),
  * forecasts 1-4 weeks ahead by simulating 1,000 paths forward, then back to admissions.

Leak safety: only the hospital data passed in (as published up to `last`) is used.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.wis import QUANTILES
from models.models import K_STEPS

P = 6
TRAIN_WEEKS = 156
N_PATHS = 1000


def _wide(hosp: pd.DataFrame, last: pd.Timestamp, pop: pd.Series) -> pd.DataFrame:
    h = hosp[(hosp["date"] <= last) & (hosp["date"] > last - pd.Timedelta(weeks=TRAIN_WEEKS + P))]
    h = h.dropna(subset=["value"])
    w = h.pivot_table(index="date", columns="location", values="value", aggfunc="last")
    w = w.reindex(pd.date_range(w.index.min(), last, freq="7D"))
    w = w[[c for c in w.columns if c in pop.index]]
    rate = w / pop[w.columns] * 1e5
    return (rate.clip(lower=0) + 0.01) ** 0.25


def ar6_forecast(hosp: pd.DataFrame, last: pd.Timestamp, pop: pd.Series, seed: int = 0) -> dict:
    """hosp: hospital data as published (date, location, value), newest week = last.
    Returns {(location, k): 23 quantiles of admissions}."""
    y = _wide(hosp, last, pop)
    X, Y, L = [], [], []
    for loc in y.columns:
        s = y[loc].values
        for t in range(P, len(s)):
            row = s[t - P:t][::-1]  # y_(t-1) ... y_(t-6)
            if np.isnan(row).any() or np.isnan(s[t]):
                continue
            X.append(row)
            Y.append(s[t])
            L.append(loc)
    X, Y, L = np.array(X), np.array(Y), np.array(L)
    A = np.column_stack([np.ones(len(X)), X])
    coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
    resid = Y - A @ coef
    pooled = resid.std(ddof=P + 1)
    sigma = {loc: (resid[L == loc].std(ddof=1) if (L == loc).sum() >= 20 else pooled) for loc in y.columns}

    rng = np.random.default_rng(seed + int(last.strftime("%Y%m%d")))
    out = {}
    for loc in y.columns:
        hist = y[loc].values[-P:][::-1]  # newest first
        if len(hist) < P or np.isnan(hist).any():
            continue
        paths = np.tile(hist, (N_PATHS, 1))  # (paths, 6), newest first
        rate_pop = pop[loc] / 1e5
        for k in range(1, max(K_STEPS) + 1):
            nxt = coef[0] + paths @ coef[1:] + rng.normal(0, sigma[loc], N_PATHS)
            nxt = np.maximum(nxt, 0.01 ** 0.25)
            paths = np.column_stack([nxt, paths[:, :-1]])
            if k in K_STEPS:
                counts = (nxt ** 4 - 0.01).clip(min=0) * rate_pop
                out[(loc, k)] = np.maximum.accumulate(np.quantile(counts, QUANTILES))
    return out
