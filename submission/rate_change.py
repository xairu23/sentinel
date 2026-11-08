"""The hub's optional 'wk flu hosp rate change' target: probabilities of 5 trend categories, derived
from Sentinel's 23 quantiles. Category rules: hub model-output/README.md, "Rate-trend forecast
specifications" (2026-27 thresholds).

The change is measured between the target week and the latest reported week (reference date - 1
week), as a rate per 100,000 people. A change is "stable" if its rate is below the stable threshold OR
the count change is below 10 admissions; otherwise it is an (large) increase or decrease depending on
the large-change threshold. Horizons 0-3.
"""
from __future__ import annotations

import numpy as np

from backtest.wis import QUANTILES

CATEGORIES = ["large_decrease", "decrease", "stable", "increase", "large_increase"]
# horizon: (stable threshold, large-change threshold), per 100k
THRESHOLDS = {0: (0.3, 1.7), 1: (0.5, 3.0), 2: (0.7, 4.0), 3: (1.0, 5.0)}


def cdf_from_quantiles(q: np.ndarray, x: float) -> float:
    """P(X <= x) from 23 quantiles: linear between quantiles; the 1% tails spread linearly down to 0
    admissions and up to one more inter-quantile step beyond the 99% quantile."""
    q = np.maximum.accumulate(np.asarray(q, float))
    q = q + np.arange(len(q)) * 1e-9  # strictly increasing (rounded quantiles can tie)
    if x < q[0]:
        return float(QUANTILES[0] * max(0.0, x) / q[0]) if q[0] > 0 else 0.0
    if x > q[-1]:
        top = q[-1] + max(q[-1] - q[-3], 1.0)
        return float(min(1.0, QUANTILES[-1] + (1 - QUANTILES[-1]) * (x - q[-1]) / (top - q[-1])))
    return float(np.interp(x, q, QUANTILES))


def boundaries(base: float, pop: float, horizon: int) -> tuple[float, float]:
    """Count-change sizes where 'stable' ends and where 'large' starts."""
    s, l = THRESHOLDS[horizon]
    b_s = max(s * pop / 1e5, 10.0)
    return b_s, max(l * pop / 1e5, b_s)


def category(base: float, value: float, pop: float, horizon: int) -> str:
    """The observed category (used for scoring)."""
    d = value - base
    b_s, b_l = boundaries(base, pop, horizon)
    if abs(d) < b_s:
        return "stable"
    if d > 0:
        return "large_increase" if d >= b_l else "increase"
    return "large_decrease" if -d >= b_l else "decrease"


def pmf(q: np.ndarray, base: float, pop: float, horizon: int) -> dict:
    b_s, b_l = boundaries(base, pop, horizon)
    F = lambda x: cdf_from_quantiles(q, x)
    # admissions are whole numbers, so "stable" means |change| <= ceil(b_s) - 1 and "large" means
    # |change| >= ceil(b_l): cut the continuous distribution half-way between those whole numbers
    # (otherwise a change of exactly 10 would count as stable; found by the v3 review)
    s_edge, l_edge = np.ceil(b_s) - 0.5, np.ceil(b_l) - 0.5
    cuts = [F(base - l_edge), F(base - s_edge), F(base + s_edge), F(base + l_edge)]
    p = np.diff([0.0] + cuts + [1.0])
    p = np.clip(p, 0, None)
    p = p / p.sum()
    p = np.round(p, 4)
    p[np.argmax(p)] += 1.0 - p.sum()  # exactly 1 after rounding
    return dict(zip(CATEGORIES, p.round(4)))
