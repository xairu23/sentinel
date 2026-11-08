"""Sentinel QAR: direct quantile autoregression on whitened rates (v3 candidate, idea 3 in
backtest/V3_PROTOCOL.md).

Follows the method CMU Delphi published for its "CMU-TimeSeries" FluSight entry
(FluSight-forecast-hub/model-metadata/CMU-TimeSeries.yml):
  * lagged values of the population-normalised hospital series (this week, 1 and 2 weeks ago),
  * "whitened" so that ~10 seasons of FluSurv-NET and ILI+ history (before July 2022) can be used as
    extra training rows: the same common scale as the tree model (models/gbq.py: rate per 100k ->
    fourth root -> divided by the series' own in-season 95th percentile -> centred),
  * fitted jointly across states, using only training weeks within +/-3 weeks of the forecast
    week's time of year (all past years),
  * one linear quantile regression per quantile level and horizon, sorted afterwards (no crossing).

It models the LEVEL a few weeks ahead and fits each quantile directly, where Sentinel's regressions
model log-growth and build ranges from past errors: a different kind of model, so different mistakes.

Leak safety: same as models/gbq.py (the hospital series' scale/centre come from data published up
to the forecast week; history ends before July 2022; a hospital training row is used only if its
target week is <= the forecast week).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from backtest.wis import QUANTILES
from models.gbq import GBQ, ROOT, featurize, history_frame, season_week, transform
from models.models import K_STEPS


@dataclass(frozen=True)
class QarConfig:
    lags: tuple = ("cs", "cs_l1", "cs_l2")
    half_window: int = 3            # +/- weeks of the same time of year
    use_history: bool = True
    nhsn_weight: float = 1.0        # weight of hospital (NHSN) rows vs history rows (round 3, idea 12)
    window: str = "time_of_year"    # "time_of_year" (+/- half_window weeks) or "in_season" (weeks 5-45, idea 20a/b)
    season_week_term: bool = False  # add the season week as a regressor (idea 20b)
    ed: bool = False                # ER flu share slope + ER-vs-admissions change, 0 where missing (round 33)


QAR = QarConfig()


def _circ_dist(a: np.ndarray, b: int) -> np.ndarray:
    # season weeks run 1..52 (or 53): week 53 is one week before week 1 of the next season
    d = np.abs(a - b)
    return np.minimum(d, 53 - d)


def quantile_fit_lp(X: np.ndarray, y: np.ndarray, tau: float) -> np.ndarray:
    """Exact linear quantile regression (pinball loss) as a linear program, solved with HiGHS.
    min tau*sum(u) + (1-tau)*sum(v)  s.t.  X b + u - v = y,  u, v >= 0.  (Slow; used as fallback.)"""
    from scipy.sparse import csr_matrix, eye, hstack
    n, p = X.shape
    c = np.concatenate([np.zeros(p), np.full(n, tau), np.full(n, 1 - tau)])
    A = hstack([csr_matrix(X), eye(n), -eye(n)], format="csr")
    bounds = [(None, None)] * p + [(0, None)] * (2 * n)
    r = linprog(c, A_eq=A, b_eq=y, bounds=bounds, method="highs")
    if r.status != 0:
        raise RuntimeError(f"quantile regression failed: {r.message}")
    return r.x[:p]


def quantile_fit(X: np.ndarray, y: np.ndarray, tau: float) -> np.ndarray:
    """Linear quantile regression by iteratively reweighted least squares (statsmodels QuantReg):
    ~50x faster than the exact LP and equal to it to ~1e-6 in the coefficients (checked on
    6,000-row problems; pinball losses equal to 1e-10). Falls back to the LP if it doesn't converge."""
    import warnings
    import statsmodels.api as sm
    from statsmodels.tools.sm_exceptions import IterationLimitWarning
    with warnings.catch_warnings():
        warnings.simplefilter("error", IterationLimitWarning)
        try:
            return np.asarray(sm.QuantReg(y, X).fit(q=tau, max_iter=5000, p_tol=1e-8).params)
        except (IterationLimitWarning, np.linalg.LinAlgError, ValueError):
            return quantile_fit_lp(X, y, tau)


def training_rows(allf: pd.DataFrame, last: pd.Timestamp, cfg: QarConfig) -> pd.DataFrame:
    if cfg.window == "in_season":
        return allf[allf["sw"].between(5, 45)]
    sw_now = int(season_week(pd.Series([last]))[0])
    near = _circ_dist(allf["sw"].values, sw_now) <= cfg.half_window
    return allf[near]


ED_COLS = ("ed_s3", "ed_gap_d2")


def qar_forecast(hosp: pd.DataFrame, last: pd.Timestamp, pop: pd.Series, cfg: QarConfig = QAR,
                 ed: pd.DataFrame | None = None) -> dict:
    """hosp: hospital data as published (date, location, value), newest week = last.
    Returns {(location, k): 23 quantiles of admissions}."""
    locs = pd.read_csv(ROOT / "data/clean/locations.csv", dtype={"location": str})
    pop_by_abbr = locs.set_index(locs["abbreviation"].str.upper())["population"]
    abbr_to_fips = dict(zip(locs["abbreviation"].str.upper(), locs["location"]))
    h = hosp[hosp["date"] <= last].dropna(subset=["value"])
    cur = pd.DataFrame({"series_id": "nhsn_" + h["location"], "source": "nhsn", "loc_key": h["location"],
                        "date": h["date"], "rate": h["value"] / h["location"].map(pop) * 1e5 + 0.75 ** 4,
                        "pop": h["location"].map(pop)}).dropna(subset=["rate", "pop"])
    cur = featurize(transform(cur))
    frames = [cur]
    if cfg.use_history:
        frames.append(history_frame(GBQ, pop_by_abbr, abbr_to_fips))
    allf = pd.concat(frames, ignore_index=True)
    if cfg.ed:
        from models.gbq import add_ed
        if ed is None:
            raise ValueError("cfg.ed needs the ER data (ed=...)")
        allf = add_ed(allf, ed[ed["date"] <= last])
        has = allf[list(ED_COLS)].notna().all(axis=1)
        for c in ED_COLS:
            allf[c] = allf[c].where(has, 0.0)
    test = allf[(allf["source"] == "nhsn") & (allf["date"] == last)].dropna(subset=list(cfg.lags))
    tr = training_rows(allf, last, cfg)
    out = {}
    cols = list(cfg.lags) + (["sw"] if cfg.season_week_term else []) + (list(ED_COLS) if cfg.ed else [])
    Xt = np.column_stack([np.ones(len(test)), test[cols].values])
    for k in K_STEPS:
        tgt = f"tgt{k}"
        d = tr.dropna(subset=list(cfg.lags) + [tgt])
        # a hospital row is usable only if its target week was already published
        d = d[(d["source"] != "nhsn") | (d["date"] + pd.Timedelta(weeks=k) <= last)]
        X = np.column_stack([np.ones(len(d)), d[cols].values])
        y = d[tgt].values  # change on the common scale (equivalent to the level: cs is a regressor)
        if cfg.nhsn_weight != 1.0:
            # weighted pinball loss: scaling a row by w > 0 multiplies its loss by w
            wt = np.where(d["source"].values == "nhsn", cfg.nhsn_weight, 1.0)
            X, y = X * wt[:, None], y * wt
        B = np.column_stack([quantile_fit(X, y, float(t)) for t in QUANTILES])  # (p, 23)
        dq = np.sort(Xt @ B, axis=1)
        for i, r in enumerate(test.itertuples(index=False)):
            it = (r.cs + dq[i] + r.center) * (r.scale + 0.01)
            rate = np.maximum(it, 0) ** 4 - 0.01 - 0.75 ** 4
            out[(r.loc_key, k)] = np.maximum.accumulate(np.maximum(rate, 0) * r.pop / 1e5)
    return out
