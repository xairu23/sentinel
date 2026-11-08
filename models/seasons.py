"""Flu-season calendar, shared by every model.

A season starts at ISO week 31 (early August) and is named by the ISO year it starts in.
Both functions use the ISO year, not the calendar year: some years have an ISO week 53 whose
Saturday falls in January (e.g. 2027-01-02 is week 53 of 2026). Using the calendar year put that
week in the wrong season and reset "admissions so far this season" right at the winter peak
(found by the independent review of v3, October 2026).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _iso(d: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    iso = pd.to_datetime(pd.Series(d)).dt.isocalendar()
    return iso["year"].astype(int).values, iso["week"].astype(int).values


def season_of(d: pd.Series) -> np.ndarray:
    """Season label (its starting ISO year) for each date."""
    y, w = _iso(d)
    return y - (w < 31).astype(int)


def _weeks_in_iso_year(y: np.ndarray) -> np.ndarray:
    dec28 = pd.to_datetime(pd.Series(y).astype(str) + "-12-28")
    return dec28.dt.isocalendar()["week"].astype(int).values


def season_week(d: pd.Series) -> np.ndarray:
    """Week number within the season: 1 = ISO week 31, counting on through ISO week 52 or 53 into
    the next year without repeating a number."""
    y, w = _iso(d)
    start_year = y - (w < 31).astype(int)
    n = _weeks_in_iso_year(start_year)
    return np.where(w >= 31, w - 30, w + n - 30)
