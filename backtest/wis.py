"""Weighted interval score (WIS), the CDC FluSight scoring rule, computed from quantile forecasts.

WIS from quantiles = mean over quantile levels of the pinball loss, times 2:
    WIS = (2 / K) * sum_k  [ (1{y <= q_k} - tau_k) * (q_k - y) ]
This equals the interval-based WIS used by the hubs (Bracher et al. 2021). Lower is better.
"""
import numpy as np

QUANTILES = np.array([0.01, 0.025, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
                      0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.975, 0.99])


def wis(q: np.ndarray, y: float, taus: np.ndarray = QUANTILES) -> float:
    """q: predicted values at each quantile level (same order as taus); y: observed value."""
    q = np.asarray(q, dtype=float)
    pinball = ((y <= q).astype(float) - taus) * (q - y)
    return float(2.0 * pinball.mean())


def covered(q: np.ndarray, y: float, level: float, taus: np.ndarray = QUANTILES) -> bool:
    """Is y inside the central prediction interval of the given level (e.g. 0.5 or 0.95)?"""
    lo = np.interp((1 - level) / 2, taus, q)
    hi = np.interp(1 - (1 - level) / 2, taus, q)
    return bool(lo <= y <= hi)
