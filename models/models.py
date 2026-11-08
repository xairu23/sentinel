"""Sentinel forecasting models. Each returns quantile forecasts of weekly admissions.

Conventions (match the FluSight hub):
- Data available at a forecast is through week `last` (the Saturday before the reference date).
- Hub horizons 0..3 are the weeks ending reference_date + 7*h, i.e. 1..4 weeks after `last`.
  Internally we call that k = h + 1 steps ahead.

Models
1. baseline       CDC-style flat baseline: median = last observed value, spread from that state's past
                  week-to-week changes (symmetrised), widening with the horizon. The bar to beat.
2. growth family  pooled regression on the log scale: future log change ~ recent log changes of admissions,
                  time of year, and the current level relative to the state's usual level.
                  Optional extra signals ("signals"):
                    "ww" = state wastewater level (CDC NWSS influenza A) and its recent changes
                    "ed" = share of emergency-department visits that are flu (CDC NSSP) and its recent changes
                  growth (no signal), growth_ww, growth_ed, growth_ed_ww.
                  The research experiment compares models that differ ONLY by the wastewater signal.
Intervals come from the empirical quantiles of each regression's past residuals.
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingRegressor

from backtest.wis import QUANTILES

K_STEPS = (1, 2, 3, 4)  # steps ahead of the last observed week = hub horizons 0..3
BASE_COLS = ["d1", "d2", "d3", "sin", "cos", "rel", "d1sq", "d1_rel"]
import itertools

# Pre-specified signals: every combination of these three is fit and goes in the submitted ensemble.
# ed and ww were pre-specified from the start; weather earned its place via Experiment 3 (see
# HOW_IT_WORKS.md) -- its 95% interval was entirely below 0 both alone and alongside ed+ww.
ENSEMBLE_SIGNALS = ("ed", "ww", "weather")
MODELS = {}
for r in range(len(ENSEMBLE_SIGNALS) + 1):
    for combo in itertools.combinations(ENSEMBLE_SIGNALS, r):
        name = "growth" + ("_" + "_".join(combo) if combo else "")
        MODELS[name] = combo
ENSEMBLE_MEMBERS = tuple(MODELS.keys())  # all 8 combinations of ed / ww / weather

# Experimental signals: fit and scored (so they can be compared fairly) but NOT averaged into the
# submitted ensemble unless an experiment shows a real improvement and ENSEMBLE_SIGNALS is updated.
MODELS.update({
    "growth_nb": ("nb",),                          # Experiment 1: neighbouring states -- no clear effect
    "growth_ed_ww_nb": ("ed", "ww", "nb"),
    "growth_lab": ("lab",),                        # Experiment 2: lab-test positivity -- redundant w/ ed+ww
    "growth_ed_ww_lab": ("ed", "ww", "lab"),
    "growth_ed_ww_nb_lab": ("ed", "ww", "nb", "lab"),
})


# ---------- 1. baseline ----------------------------------------------------------------------
def baseline(series: pd.Series, k: int) -> np.ndarray:
    """series: admissions for one location up to and including `last`, indexed by date."""
    y = series.values.astype(float)
    last = y[-1]
    diffs = np.diff(y)
    diffs = np.concatenate([diffs, -diffs])  # symmetric
    spread = np.quantile(diffs, QUANTILES) * np.sqrt(k)
    q = np.maximum(last + spread, 0)
    q[QUANTILES == 0.5] = last
    return np.maximum.accumulate(q)


# ---------- 2. pooled log-growth regression --------------------------------------------------
def make_features(panel: pd.DataFrame) -> pd.DataFrame:
    """panel: columns date, location, value and optionally ww_level, ed_level. Adds all features."""
    p = panel.sort_values(["location", "date"]).copy()
    p["ly"] = np.log1p(p["value"])
    g = p.groupby("location")["ly"]
    p["d1"] = g.diff(1)              # change over the last week
    p["d2"] = g.diff(2) - p["d1"]    # change in the week before that
    p["d3"] = g.diff(3) - g.diff(2)  # change three weeks ago
    # time of year (flu rises in late autumn, peaks in winter, falls in spring)
    wk = p["date"].dt.isocalendar().week.astype(float)
    p["sin"] = np.sin(2 * np.pi * wk / 52.18)
    p["cos"] = np.cos(2 * np.pi * wk / 52.18)
    # how high the state is compared with its own recent 2-year typical level (mean reversion)
    p["rel"] = p["ly"] - g.transform(lambda s: s.rolling(104, min_periods=30).median().shift(1))
    # near-peak dampening: plain linear regression on d1 extrapolates growth forever, so a very
    # sharp rise gets a forecast that keeps rising -- even once the real curve has already turned
    # over (see Experiment 4 in HOW_IT_WORKS.md, the Jan 2026 whiplash). These let the model learn
    # nonlinear mean-reversion: d1sq lets a big rise curve back down instead of extrapolating in a
    # straight line; d1_rel flags "fast growth while already far above the usual level", which is
    # exactly what a peak looks like just before it turns.
    p["d1sq"] = np.sign(p["d1"]) * p["d1"] ** 2
    p["d1_rel"] = p["d1"] * p["rel"]
    if NEIGHBOR_PAIRS:
        p = add_neighbor_features(p)
    for sig in ("ww", "ed", "lab", "weather"):
        col = f"{sig}_level"
        if col in p:
            gs = p.groupby("location")[col]
            p[f"{sig}1"] = gs.diff(1)
            p[f"{sig}2"] = gs.diff(2) - p[f"{sig}1"]
            p[f"{sig}l"] = p[col]
    return p


def feature_cols(signals=()) -> list[str]:
    cols = list(BASE_COLS)
    for sig in signals:
        if sig == "nb":
            cols += ["nb1", "nb2"]
        else:
            cols += [f"{sig}1", f"{sig}2", f"{sig}l"]
    return cols


# ---------- neighbouring states --------------------------------------------------------------
NEIGHBOR_PAIRS: list = []  # filled by load_neighbor_pairs(); list of (fips, neighbour_fips)


def load_neighbor_pairs(locations_csv) -> None:
    import pandas as _pd
    from models.neighbors import neighbor_pairs
    NEIGHBOR_PAIRS[:] = neighbor_pairs(_pd.read_csv(locations_csv, dtype={"location": str}))


def add_neighbor_features(p: pd.DataFrame) -> pd.DataFrame:
    """nb1 / nb2 = average of the neighbours' last-week and previous-week changes (same week, same
    data version as the state itself, so nothing from the future is used)."""
    pairs = pd.DataFrame(NEIGHBOR_PAIRS, columns=["location", "nb"])
    nb = p[["date", "location", "d1", "d2"]].rename(columns={"location": "nb", "d1": "nb1", "d2": "nb2"})
    m = pairs.merge(nb, on="nb").groupby(["location", "date"])[["nb1", "nb2"]].mean().reset_index()
    return p.merge(m, on=["location", "date"], how="left")


VOL_WINDOW = 8     # weeks of recent history used to gauge how volatile growth has been right now
VOL_CAP = 3.0       # don't widen intervals more than 3x even if recent swings are extreme
BIAS_WINDOW = 6     # weeks of recent history used to check if the model's been running low or high
BIAS_CAP = 1.0      # clip the correction to at most ~x2.7 (exp(1)) either way, so it can't run away
BIAS_SHRINK = 0.3   # only apply this fraction of the measured recent bias -- a full-strength nudge
                     # chases short-term noise and hurts far more often than it helps (tested: a
                     # naive version made the historical backtest much worse, see HOW_IT_WORKS.md)


def fit_and_predict(feat: pd.DataFrame, last: pd.Timestamp, signals=(), train_weeks: int = 156) -> dict:
    """Train on rows whose target is known by `last`; predict k steps ahead from `last` for every location.
    Returns {(location, k): array of 23 quantiles}.

    Prediction intervals are widened per-location when that state's growth rate has been more
    volatile than usual recently (computed from data up to and including `last`, so nothing from
    the future is used): a model trained on 3 years of "typical" ups and downs has intervals too
    narrow for a sudden, unusually sharp swing, and gets penalised heavily (worse than even the
    naive baseline) exactly when it matters most. See Experiment 4 in HOW_IT_WORKS.md."""
    cols = feature_cols(signals)
    out = {}
    p = feat.copy()
    for k in K_STEPS:
        p["target"] = p.groupby("location")["ly"].shift(-k) - p["ly"]
        # target date must be <= last (no peeking), keep a rolling window of recent seasons
        p["target_date"] = p["date"] + pd.Timedelta(weeks=k)
        train = p[(p["target_date"] <= last) & (p["date"] > last - pd.Timedelta(weeks=train_weeks))]
        train = train.replace([np.inf, -np.inf], np.nan).dropna(subset=cols + ["target"])
        if len(train) < 200:
            continue
        X = sm.add_constant(train[cols], has_constant="add")
        model = sm.OLS(train["target"], X).fit()
        resid_q = np.quantile(model.resid, QUANTILES)
        median_resid = np.median(model.resid)
        pop_vol = train["d1"].std()  # typical week-to-week growth volatility across the training set

        # each location's own recent volatility (data through `last` only -- no peeking)
        recent = p[(p["date"] <= last) & (p["date"] > last - pd.Timedelta(weeks=VOL_WINDOW))]
        recent_vol = recent.groupby("location")["d1"].std()

        # recent-bias correction: a trend-following model lags behind any sharp CHANGE in trend --
        # it undershoots at the start of a new season's climb just as it overshoots at a sudden
        # peak (see Experiment 4 and Experiment 5 in HOW_IT_WORKS.md). Checking whether the model's
        # own recent guesses (on data it's already seen, up to `last`) have been running low or
        # high, and nudging the current forecast by that amount, catches a persistent lag quickly
        # instead of waiting 3 years of training data to "learn" it.
        train_resid = train.assign(resid=model.resid.values)
        bias_recent = train_resid[train_resid["date"] > last - pd.Timedelta(weeks=BIAS_WINDOW)]
        per_loc_bias = bias_recent.groupby("location")["resid"].mean()
        global_bias = bias_recent["resid"].mean() if len(bias_recent) else 0.0

        now = p[p["date"] == last].replace([np.inf, -np.inf], np.nan).dropna(subset=cols)
        Xn = sm.add_constant(now[cols], has_constant="add")
        mu = model.predict(Xn)
        for (loc, ly0), m in zip(now[["location", "ly"]].itertuples(index=False), mu):
            vol_ratio = 1.0
            if pop_vol and pop_vol > 0 and loc in recent_vol.index and pd.notna(recent_vol[loc]):
                vol_ratio = np.clip(recent_vol[loc] / pop_vol, 1.0, VOL_CAP)
            scaled_resid = median_resid + (resid_q - median_resid) * vol_ratio
            bias = per_loc_bias[loc] if loc in per_loc_bias.index and pd.notna(per_loc_bias[loc]) else global_bias
            bias = np.clip((bias if pd.notna(bias) else 0.0) * BIAS_SHRINK, -BIAS_CAP, BIAS_CAP)
            # cap the predicted log change (a k-week change beyond about x7.4 is implausible)
            q = np.expm1(ly0 + np.clip(m + bias + scaled_resid, -2.0, 2.0))
            out[(loc, k)] = np.maximum.accumulate(np.maximum(q, 0))
    return out


# A coarser quantile grid for the tree model -- fitting one boosted model PER quantile level is
# expensive (23 separate model fits per horizon), so fit a handful and fill in the rest by
# interpolation between them. The 23-level grid is still what gets scored and submitted.
GBM_TAUS = np.array([0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975])


def fit_and_predict_gbm(feat: pd.DataFrame, last: pd.Timestamp, signals=(), train_weeks: int = 156) -> dict:
    """Same forecasting job as fit_and_predict, but with gradient-boosted decision trees instead of
    linear regression. Trees can learn nonlinear patterns (like 'a very fast rise, especially when
    already far above normal, tends to bend back down') and interactions between features
    automatically, instead of needing them hand-built in like d1sq / d1_rel. A genuinely different
    model type (not just another linear variant) catches different mistakes -- see Experiment 6 in
    HOW_IT_WORKS.md."""
    cols = feature_cols(signals)
    out = {}
    p = feat.copy()
    for k in K_STEPS:
        p["target"] = p.groupby("location")["ly"].shift(-k) - p["ly"]
        p["target_date"] = p["date"] + pd.Timedelta(weeks=k)
        train = p[(p["target_date"] <= last) & (p["date"] > last - pd.Timedelta(weeks=train_weeks))]
        train = train.replace([np.inf, -np.inf], np.nan).dropna(subset=cols + ["target"])
        if len(train) < 200:
            continue
        now = p[p["date"] == last].replace([np.inf, -np.inf], np.nan).dropna(subset=cols)
        if len(now) == 0:
            continue
        Xtr, ytr, Xn = train[cols].values, train["target"].values, now[cols].values
        preds_by_tau = {}
        for tau in GBM_TAUS:
            m = HistGradientBoostingRegressor(loss="quantile", quantile=float(tau),
                                               max_depth=4, max_iter=150, learning_rate=0.08,
                                               min_samples_leaf=20, random_state=0)
            m.fit(Xtr, ytr)
            preds_by_tau[tau] = m.predict(Xn)
        for i, (loc, ly0) in enumerate(now[["location", "ly"]].itertuples(index=False)):
            coarse = np.array([preds_by_tau[tau][i] for tau in GBM_TAUS])
            coarse = np.sort(coarse)  # enforce monotonic quantiles before interpolating
            fine = np.interp(QUANTILES, GBM_TAUS, coarse)
            q = np.expm1(ly0 + np.clip(fine, -2.0, 2.0))
            out[(loc, k)] = np.maximum.accumulate(np.maximum(q, 0))
    return out


def run_all(feat: pd.DataFrame, last: pd.Timestamp) -> dict:
    """All growth-family models -> {model_name: {(location, k): quantiles}}."""
    out = {name: fit_and_predict(feat, last, signals) for name, signals in MODELS.items()}
    out["gbm_ed_ww_weather"] = fit_and_predict_gbm(feat, last, ("ed", "ww", "weather"))
    return out


def ensemble(preds: dict, loc: str, k: int) -> np.ndarray | None:
    """Sentinel ensemble: equal-weight average of the quantiles of every growth model available for
    this location (pre-specified rule: all four variants; fewer where a signal is missing)."""
    qs = [preds[m][(loc, k)] for m in ENSEMBLE_MEMBERS if m in preds and (loc, k) in preds[m]]
    if not qs:
        return None
    return np.maximum.accumulate(np.maximum(np.mean(qs, axis=0), 0))
