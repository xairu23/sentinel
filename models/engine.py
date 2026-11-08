"""Sentinel forecasting engine (v2). ONE implementation used by both the backtest and the live
weekly forecast, so what gets tested is exactly what gets submitted.

What it does, given the hospital data as published this week:
 1. Reporting-delay fix: the newest week is usually revised UP later (median about +5%, more in
    recent seasons). Learn by how much from earlier published versions (past only) and correct it.
 2. Attach the extra signals: ER visits for the SAME newest week (CDC publishes them the same day),
    wastewater one week behind (real reporting lag), weather (same week).
 3. Features: recent log-growth (3 lags), time of year, level vs. the state's usual level,
    peak-dampening terms, and cumulative admissions so far this season per 100k people
    ("how many have already been infected" -- the main reason an epidemic turns over).
 4. Eight pooled regressions (every combination of ER / wastewater / weather), trained on the last
    three years with RECENT weeks counting more (half-life 52 weeks), one per horizon.
 5. Intervals from each regression's past errors, widened when the state has been jumpy, plus a
    small (30%) correction for the model's own recent bias. Average the eight.

Every choice above was decided on the 2023-24 and 2024-25 seasons only; see HOW_IT_WORKS.md.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd
import statsmodels.api as sm

from backtest.wis import QUANTILES
from models.models import BASE_COLS, K_STEPS, make_features
from models.seasons import season_of


@dataclass(frozen=True)
class Config:
    signals: tuple = ("ed", "ww", "weather")
    extra_cols: tuple = ("cum", "cum_d1")
    train_weeks: int = 156
    recency_halflife: float = 52.0
    vol_scale: bool = True
    bias_shrink: float = 0.3
    bias_window: int = 6
    backfill: int = 1
    backfill_m: int = 10
    spread: float = 1.0
    resid_bins: int = 0
    ww_delay: int = 1
    ed_delay: int = 0
    weather_delay: int = 0
    hist: bool = False             # also learn growth dynamics from ~10 seasons of pre-2022 history
    hist_sources: tuple = ("flusurv", "iliplus")
    hist_weight: float = 0.3
    hist_min: float = 0.5
    loc_scale: float = 0.0         # 0 = pooled error spread; 1 = each location's own error spread
    gbq_weight: float = 0.0        # v2: weight of the history-trained tree model (models/gbq.py) in the blend
    gbq_bags: int = 3
    families: tuple = ()           # v3: equal-weight quantile average of the regressions and these
                                   # model families ("gbq" = trees, "qar" = quantile autoregression)
    qar_skip: tuple = ()           # places where QAR's pooled mainland seasonality doesn't apply
    trees_corrected: bool = False  # v3.1: the trees get the reporting-delay-corrected newest week too
    qar_tail: tuple = (1.0, 1.0)   # v3.2: QAR's lower / upper half pulled toward its median (log(x+1) scale)
    er_anchor: tuple = ()          # v3.3: (a, T, gate) ER-visit anchor for places under T/week, see er_anchor()
    gbq_spatial: bool = False      # trees also see neighbouring states' level and slope (idea 18, round 28)
    gbq_context: bool = False      # trees also see season position and the national level/slope (round 31)
    gbq_ed: bool = False           # trees also see the ER flu share level and slopes (round 32)


# the 8 regressions, with each location's ranges half-way adjusted to its own error size (Exp. 11)
LINEAR = Config(loc_scale=0.5)
# v2: the regressions blended 67/33 with the history-trained tree model (Experiment 10)
V2 = Config(loc_scale=0.5, gbq_weight=0.33, gbq_bags=6)
# v3: equal-weight average of three model families: regressions, trees, quantile autoregression
#     (backtest/V3_PROTOCOL.md, idea 7)
V3 = Config(loc_scale=0.5, gbq_bags=6, families=("gbq", "qar"), qar_skip=("72",))  # 72 = Puerto Rico (idea 9)
# v3.1: as v3, but the trees see the newest week after the reporting-delay fix, like the regressions
#       (backtest/V3_PROTOCOL.md, idea 19, trees part)
V31 = Config(loc_scale=0.5, gbq_bags=6, families=("gbq", "qar"), qar_skip=("72",), trees_corrected=True)
# v3.2 (round 24, backtest/V3_PROTOCOL.md): QAR's lower tail was far too long (truth below its 2.5% line
# 0.2% of the time); its lower half is pulled to 40% of its distance from QAR's median before averaging
V32 = Config(loc_scale=0.5, gbq_bags=6, families=("gbq", "qar"), qar_skip=("72",), trees_corrected=True,
             qar_tail=(0.4, 1.0))
# v3.3 (round 26c): ER-visit anchor -- in places under 200/week whose newest ER flu share is >= 1%, move the
# whole forecast half-way (log scale) toward the newest week implied by the ER share
V33 = replace(V32, er_anchor=(0.5, 200, 0.01))
# v3.4 (round 28): v3.3 with the trees also seeing neighbouring states' level and slope
V34 = replace(V33, gbq_spatial=True)
# v3.5 (round 31): v3.4 with the trees also seeing where the season stands and what the whole country is doing
V35 = replace(V34, gbq_context=True)
# v3.6 (round 32): v3.5 with the trees also seeing flu ER visits
V36 = replace(V35, gbq_ed=True)
FINAL = V36  # what the weekly forecast submits
ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]


# ---------- long history (optional) ---------------------------------------------------------------
def load_history(cfg: Config = FINAL) -> pd.DataFrame:
    """FluSurv-NET hospitalisation rates and ILI+ by state, BEFORE July 2022 only (see
    scripts/build_history.py), turned into the same base features as the hospital data.
    Rescaled to state-sized pseudo-counts so log1p behaves like it does on admissions."""
    from scripts.lab_clean import epiweek_to_saturday
    h = pd.read_csv(ROOT / "data/history/sentinel_history.csv")
    h["date"] = epiweek_to_saturday(h["epiweek"])
    h = h[h["date"] < "2022-07-01"]
    loc = pd.read_csv(ROOT / "data/clean/locations.csv", dtype={"location": str})
    pop_by_abbr = loc.set_index(loc["abbreviation"].str.upper())["population"]
    out = []
    if "flusurv" in cfg.hist_sources:
        f = h[h["source"] == "flusurv"].copy()
        f["rate"] = f["v1"]
        f["value"] = f["rate"] * f["location"].str.split("_").str[0].str.upper().map(pop_by_abbr) / 1e5
        f["id"] = "fs_" + f["location"]
        out.append(f[["id", "date", "value", "rate"]])
    if "iliplus" in cfg.hist_sources:
        g = h[h["source"] == "iliplus"].copy()
        g["rate"] = g["v1"] * g["v2"] / 100 * 4  # roughly admissions-per-100k scale at the peak
        g["value"] = g["rate"] * g["location"].str.upper().map(pop_by_abbr) / 1e5
        g["id"] = "ip_" + g["location"]
        out.append(g[["id", "date", "value", "rate"]])
    d = pd.concat(out).dropna()
    rows = []
    for i, s in d.groupby("id"):
        s = s.set_index("date").sort_index()
        s = s[~s.index.duplicated()]
        s = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="7D"))  # gaps stay empty
        s.index.name = "date"
        rows.append(s.reset_index().assign(location=i))
    p = pd.concat(rows, ignore_index=True)
    p["ly"] = np.log1p(p["value"])
    g = p.groupby("location")["ly"]
    p["d1"] = g.diff(1)
    p["d2"] = g.diff(2) - p["d1"]
    p["d3"] = g.diff(3) - g.diff(2)
    wk = p["date"].dt.isocalendar().week.astype(float)
    p["sin"] = np.sin(2 * np.pi * wk / 52.18)
    p["cos"] = np.cos(2 * np.pi * wk / 52.18)
    p["rel"] = p["ly"] - g.transform(lambda s: s.rolling(104, min_periods=30).median().shift(1))
    p["d1sq"] = np.sign(p["d1"]) * p["d1"] ** 2
    p["d1_rel"] = p["d1"] * p["rel"]
    if "cum" in cfg.extra_cols:
        season = pd.Series(season_of(p["date"]), index=p.index)
        p["cum"] = np.log1p(p.assign(_s=season.values).groupby(["location", "_s"])["rate"].cumsum().fillna(0))
        if "cum_d1" in cfg.extra_cols:
            p["cum_d1"] = p["cum"] * p["d1"]
    return p


def history_rows(hist: pd.DataFrame, cols: list, k: int, cfg: Config = FINAL) -> pd.DataFrame:
    tgt = hist.groupby("location")["ly"].shift(-k) - hist["ly"]
    t = hist.assign(target=tgt)
    base = [c for c in cols if c in t.columns]
    t = t.replace([np.inf, -np.inf], np.nan).dropna(subset=base + ["target"])
    if cfg.hist_min:
        t = t[t["rate"] >= cfg.hist_min]
    return t


# ---------- 1. reporting-delay (backfill) correction ----------------------------------------------
def lag_table(vintages: pd.DataFrame) -> pd.DataFrame:
    """First published value of each (location, week) at each lag (0 = it was the newest week)."""
    t = vintages
    maxd = t.groupby("as_of")["date"].max().rename("maxd")
    tt = t.merge(maxd, left_on="as_of", right_index=True)
    tt["lag"] = (tt["maxd"] - tt["date"]).dt.days // 7
    first = (tt[tt["lag"] <= 2].sort_values("as_of")
             .groupby(["location", "date", "lag"], as_index=False).first())
    return first[["location", "date", "lag", "as_of", "value"]]


def backfill_correct(hosp: pd.DataFrame, last: pd.Timestamp, as_of: pd.Timestamp,
                     lags: pd.DataFrame, cfg: Config = FINAL) -> pd.DataFrame:
    """Scale up the newest `cfg.backfill` weeks by how much such weeks were revised before.
    Uses only versions published BEFORE `as_of`, compared with this version's values for weeks that
    are now >= 4 weeks old (essentially final)."""
    hosp = hosp.copy()
    cur = hosp.set_index(["location", "date"])["value"]
    for lag in range(cfg.backfill):
        f = lags[(lags["lag"] == lag) & (lags["as_of"] < as_of)
                 & (lags["date"] <= last - pd.Timedelta(weeks=4))]
        recent_dates = np.sort(f["date"].unique())[-cfg.backfill_m:]
        f = f[f["date"].isin(recent_dates)]
        c = cur.reindex(pd.MultiIndex.from_frame(f[["location", "date"]])).values
        ok = (f["value"].values > 0) & (c >= 5)
        lr = pd.Series(np.log(c[ok] / f["value"].values[ok]), index=f["location"].values[ok])
        pooled = lr.median() if len(lr) else 0.0
        per = lr.groupby(level=0).median()
        n = lr.groupby(level=0).size()
        a = n / (n + 5)  # shrink each state's own ratio toward the national one
        est = (a * per + (1 - a) * pooled).clip(-0.2, 0.5)
        m = hosp["date"] == last - pd.Timedelta(weeks=lag)
        hosp.loc[m, "value"] = hosp.loc[m, "value"] * np.exp(
            hosp.loc[m, "location"].map(est).fillna(pooled).values)
    return hosp


# ---------- 2. signals -----------------------------------------------------------------------------
def attach(hosp: pd.DataFrame, sig: dict, cfg: Config = FINAL, states_only: bool = False) -> pd.DataFrame:
    """Merge the signals onto the hospital panel with each one's real publication delay."""
    p = hosp[["date", "location", "value"]].copy()
    for name, delay in (("ww", cfg.ww_delay), ("ed", cfg.ed_delay), ("weather", cfg.weather_delay)):
        if name not in sig:
            continue
        col = f"{name}_level"
        p = p.merge(sig[name][["date", "location", col]], on=["date", "location"], how="left")
        p = p.sort_values(["location", "date"])
        # delay FIRST, then fill short gaps: filling first could pull a not-yet-published value into
        # a gap that the delay then moves into the forecast row (found by the independent review)
        p[col] = p.groupby("location")[col].shift(delay)
        p[col] = p.groupby("location")[col].transform(lambda s: s.interpolate(limit=2, limit_direction="forward"))
    if states_only:
        p = p[~p["location"].isin(["US", "11", "72"])]
    return p


# ---------- 3. features -----------------------------------------------------------------------------
_POP = {}


def add_extra(p: pd.DataFrame, extra: tuple, pop: pd.Series) -> pd.DataFrame:
    if not extra:
        return p
    p = p.copy()
    if "cum" in extra:
        # cumulative admissions so far this season (season starts week 31, early August) per 100k
        season = pd.Series(season_of(p["date"]), index=p.index)
        cs = p.assign(_s=season.values).groupby(["location", "_s"])["value"].cumsum()
        p["cum"] = np.log1p(cs / p["location"].map(pop) * 1e5)
    if "cum_d1" in extra:
        p["cum_d1"] = p["cum"] * p["d1"]
    return p


def build_features(panel: pd.DataFrame, pop: pd.Series, cfg: Config = FINAL) -> pd.DataFrame:
    return add_extra(make_features(panel), cfg.extra_cols, pop)


# ---------- 4-5. models ------------------------------------------------------------------------------
def loc_spread(train_loc: pd.Series, resid: np.ndarray, power: float) -> dict:
    """Each location's own error size relative to everyone's (robust: median absolute error),
    raised to `power` (0 = ignore, 1 = fully own). Big, smooth series (the US total, California)
    get narrower ranges; small noisy states get wider ones."""
    if not power:
        return {}
    a = np.abs(resid - np.median(resid))
    allm = np.median(a)
    per = pd.Series(a).groupby(train_loc.values).median()
    n = pd.Series(a).groupby(train_loc.values).size()
    ratio = (per / allm).clip(0.25, 4.0)
    ratio = ratio.where(n >= 20, 1.0)
    return (ratio ** power).to_dict()

def member_cols(combo: tuple, cfg: Config = FINAL) -> list:
    cols = list(BASE_COLS) + list(cfg.extra_cols)
    for s in combo:
        cols += [f"{s}1", f"{s}2", f"{s}l"]
    return cols


def members(cfg: Config = FINAL) -> list[tuple]:
    return [c for r in range(len(cfg.signals) + 1) for c in itertools.combinations(cfg.signals, r)]


def fit_member(feat: pd.DataFrame, last: pd.Timestamp, cols: list, cfg: Config = FINAL,
               hist: pd.DataFrame | None = None) -> dict:
    if hist is not None:
        return fit_member_hist(feat, last, cols, cfg, hist)
    out = {}
    p = feat
    for k in K_STEPS:
        tgt = p.groupby("location")["ly"].shift(-k) - p["ly"]
        tdate = p["date"] + pd.Timedelta(weeks=k)
        sel = (tdate <= last) & (p["date"] > last - pd.Timedelta(weeks=cfg.train_weeks))
        keep = list(dict.fromkeys(cols + ["location", "date", "value", "d1"]))
        train = p.loc[sel, keep].assign(target=tgt[sel])
        train = train.replace([np.inf, -np.inf], np.nan).dropna(subset=cols + ["target"])
        if len(train) < 200:
            continue
        now = p[p["date"] == last].replace([np.inf, -np.inf], np.nan).dropna(subset=cols)
        if now.empty:
            continue
        X = sm.add_constant(train[cols].values, has_constant="add")
        Xn = sm.add_constant(now[cols].values, has_constant="add")
        if cfg.recency_halflife:
            age = (last - train["date"]).dt.days.values / 7.0
            model = sm.WLS(train["target"].values, X, weights=0.5 ** (age / cfg.recency_halflife)).fit()
        else:
            model = sm.OLS(train["target"].values, X).fit()
        resid = train["target"].values - model.predict(X)
        resid_q = np.quantile(resid, QUANTILES)
        med_r = np.median(resid)
        bin_q = {}
        if cfg.resid_bins:
            lyt = np.log1p(train["value"].values)
            edges = np.quantile(lyt, np.linspace(0, 1, cfg.resid_bins + 1)[1:-1])
            bt = np.searchsorted(edges, lyt)
            bin_q = {b: (np.quantile(resid[bt == b], QUANTILES), np.median(resid[bt == b]))
                     for b in range(cfg.resid_bins) if (bt == b).sum() > 50}
        ls = loc_spread(train["location"], resid, cfg.loc_scale)
        pop_vol = train["d1"].std()
        recent = p[(p["date"] <= last) & (p["date"] > last - pd.Timedelta(weeks=8))]
        recent_vol = recent.groupby("location")["d1"].std()
        tb = train.assign(resid=resid)
        br = tb[tb["date"] > last - pd.Timedelta(weeks=cfg.bias_window)]
        per_b = br.groupby("location")["resid"].mean()
        glob_b = br["resid"].mean() if len(br) else 0.0
        mu = model.predict(Xn)
        for (loc, ly0), m in zip(now[["location", "ly"]].itertuples(index=False), mu):
            vr = 1.0
            if cfg.vol_scale and pop_vol > 0 and loc in recent_vol.index and pd.notna(recent_vol[loc]):
                vr = np.clip(recent_vol[loc] / pop_vol, 1.0, 3.0)
            rq, mr = resid_q, med_r
            if bin_q:
                b = int(np.searchsorted(edges, ly0))
                rq, mr = bin_q.get(b, (rq, mr))
            sr = mr + (rq - mr) * vr * cfg.spread * ls.get(loc, 1.0)
            b = 0.0
            if cfg.bias_shrink:
                b = per_b.get(loc, glob_b)
                b = np.clip((b if pd.notna(b) else 0.0) * cfg.bias_shrink, -1, 1)
            q = np.expm1(ly0 + np.clip(m + b + sr, -2.0, 2.0))
            out[(loc, k)] = np.maximum.accumulate(np.maximum(q, 0))
    return out


def fit_member_hist(feat: pd.DataFrame, last: pd.Timestamp, cols: list, cfg: Config,
                    hist: pd.DataFrame) -> dict:
    """Like fit_member, but the regression also learns from the long history. Columns the history
    doesn't have (ER, wastewater, weather) are set to 0 there with an `is_hist` indicator, so their
    effects come from recent data only. Recent rows keep their recency weights; historical rows get
    a fixed weight (cfg.hist_weight). Interval widths come from RECENT errors only."""
    out = {}
    p = feat
    for k in K_STEPS:
        tgt = p.groupby("location")["ly"].shift(-k) - p["ly"]
        tdate = p["date"] + pd.Timedelta(weeks=k)
        sel = (tdate <= last) & (p["date"] > last - pd.Timedelta(weeks=cfg.train_weeks))
        keep = list(dict.fromkeys(cols + ["location", "date", "value", "d1"]))
        train = p.loc[sel, keep].assign(target=tgt[sel])
        train = train.replace([np.inf, -np.inf], np.nan).dropna(subset=cols + ["target"])
        if len(train) < 200:
            continue
        now = p[p["date"] == last].replace([np.inf, -np.inf], np.nan).dropna(subset=cols)
        if now.empty:
            continue
        H = history_rows(hist, cols, k, cfg)
        Hx = pd.DataFrame({c: (H[c].values if c in H.columns else 0.0) for c in cols}, index=H.index)
        Xc = np.column_stack([train[cols].values, np.zeros(len(train))])
        Xh = np.column_stack([Hx.values, np.ones(len(Hx))])
        X = sm.add_constant(np.vstack([Xc, Xh]), has_constant="add")
        y = np.concatenate([train["target"].values, H["target"].values])
        if cfg.recency_halflife:
            w_cur = 0.5 ** ((last - train["date"]).dt.days.values / 7.0 / cfg.recency_halflife)
        else:
            w_cur = np.ones(len(train))
        model = sm.WLS(y, X, weights=np.concatenate([w_cur, np.full(len(Hx), cfg.hist_weight)])).fit()
        Xc1 = sm.add_constant(Xc, has_constant="add")
        resid = train["target"].values - model.predict(Xc1)
        resid_q = np.quantile(resid, QUANTILES)
        med_r = np.median(resid)
        ls = loc_spread(train["location"], resid, cfg.loc_scale)
        pop_vol = train["d1"].std()
        recent = p[(p["date"] <= last) & (p["date"] > last - pd.Timedelta(weeks=8))]
        recent_vol = recent.groupby("location")["d1"].std()
        tb = train.assign(resid=resid)
        br = tb[tb["date"] > last - pd.Timedelta(weeks=cfg.bias_window)]
        per_b = br.groupby("location")["resid"].mean()
        glob_b = br["resid"].mean() if len(br) else 0.0
        Xn = sm.add_constant(np.column_stack([now[cols].values, np.zeros(len(now))]), has_constant="add")
        mu = model.predict(Xn)
        for (loc, ly0), m in zip(now[["location", "ly"]].itertuples(index=False), mu):
            vr = 1.0
            if cfg.vol_scale and pop_vol > 0 and loc in recent_vol.index and pd.notna(recent_vol[loc]):
                vr = np.clip(recent_vol[loc] / pop_vol, 1.0, 3.0)
            sr = med_r + (resid_q - med_r) * vr * cfg.spread * ls.get(loc, 1.0)
            b = 0.0
            if cfg.bias_shrink:
                b = per_b.get(loc, glob_b)
                b = np.clip((b if pd.notna(b) else 0.0) * cfg.bias_shrink, -1, 1)
            q = np.expm1(ly0 + np.clip(m + b + sr, -2.0, 2.0))
            out[(loc, k)] = np.maximum.accumulate(np.maximum(q, 0))
    return out


def forecast_from_features(feat: pd.DataFrame, last: pd.Timestamp, cfg: Config = FINAL,
                           hist: pd.DataFrame | None = None) -> dict:
    """Average of every member's quantiles -> {(location, k): 23 quantiles}."""
    preds = [fit_member(feat, last, member_cols(c, cfg), cfg, hist) for c in members(cfg)]
    out = {}
    for key in set().union(*preds):
        qs = [p[key] for p in preds if key in p]
        out[key] = np.maximum.accumulate(np.maximum(np.mean(qs, axis=0), 0))
    return out


def prepare(hosp: pd.DataFrame, last: pd.Timestamp, as_of: pd.Timestamp, sig: dict,
            lags: pd.DataFrame | None, pop: pd.Series, cfg: Config = FINAL,
            states_only: bool = False, return_corrected: bool = False):
    """Steps 1-3 for one week: the features the regressions are fitted on, the config actually
    used (ER switched to one week behind when it wasn't published in time) and the hospital data
    as published (input of the tree model)."""
    published = hosp[hosp["date"] <= last]
    h = published
    if cfg.ed_delay == 0 and "ed" in sig and len(sig["ed"]) and sig["ed"]["date"].max() < last:
        # ER visits are used for the same week only when they were really published that way
        # (CDC's hub does so since the 2025-26 season); otherwise they're one week behind
        cfg = replace(cfg, ed_delay=1)
    if cfg.backfill and lags is not None:
        h = backfill_correct(h, last, as_of, lags, cfg)
    panel = attach(h, sig, cfg, states_only=states_only)
    if return_corrected:
        return build_features(panel, pop, cfg), cfg, published, h
    return build_features(panel, pop, cfg), cfg, published


def member_forecasts(hosp, last, as_of, sig, lags, pop, cfg: Config = FINAL, states_only=False,
                     hist=None) -> dict:
    """Each regression member's forecast separately ({member signals: {(location, k): 23 q}}),
    for experiments on how to combine them. forecast() averages exactly these."""
    feat, cfg, _ = prepare(hosp, last, as_of, sig, lags, pop, cfg, states_only)
    return {c: fit_member(feat, last, member_cols(c, cfg), cfg, hist if cfg.hist else None)
            for c in members(cfg)}


def er_anchor(preds: dict, hosp: pd.DataFrame, ed: pd.DataFrame, last: pd.Timestamp,
              a: float, T: float, gate: float) -> tuple[dict, dict]:
    """Round 26c. hosp, ed: hospital admissions and ER flu share (date, location, value) AS PUBLISHED now.
    For each place: implied = newest ER share x median(admissions / ER share) over the 4 weeks before the
    newest; d = log(1 + implied) - log(1 + newest admissions). If newest admissions < T and the newest ER
    share >= gate, every quantile is multiplied (on the x + 1 scale) by exp(a * d). Returns (preds, d used)."""
    H = hosp.set_index(["location", "date"])["value"]; E = ed.set_index(["location", "date"])["value"]
    weeks = [last - pd.Timedelta(weeks=j) for j in range(5)]
    out, used = dict(preds), {}
    for loc in {k[0] for k in preds}:
        hs = np.array([H.get((loc, w), np.nan) for w in weeks], float)
        es = np.array([E.get((loc, w), np.nan) for w in weeks], float)
        if np.isnan(hs).any() or np.isnan(es).any() or (es <= 0).any() or hs[0] >= T or es[0] < gate:
            continue
        d = np.log1p(np.median(hs[1:] / es[1:]) * es[0]) - np.log1p(hs[0]); used[loc] = d
        for k in [k for k in preds if k[0] == loc]:
            out[k] = np.expm1(np.log1p(np.maximum(np.asarray(preds[k], float), 0)) + a * d)
    return out, used


def qar_tail(q, fl: float, fu: float) -> np.ndarray:
    """Pull the lower / upper half of 23 quantiles toward the median by fl / fu on the log(x+1) scale."""
    L = np.log1p(np.maximum(np.asarray(q, float), 0)); m = L[11]; d = L - m
    return np.expm1(m + np.where(d < 0, fl, fu) * d)


def forecast(hosp: pd.DataFrame, last: pd.Timestamp, as_of: pd.Timestamp, sig: dict,
             lags: pd.DataFrame | None, pop: pd.Series, cfg: Config = FINAL,
             states_only: bool = False, hist: pd.DataFrame | None = None) -> dict:
    """The whole pipeline for one week. hosp = hospital data as published at `as_of` (newest week =
    `last`); sig = signals ({'ww','ed','weather'} -> date, location, <name>_level), already limited
    to what was published by then; lags = lag_table() of versions published before `as_of`;
    hist = load_history(cfg) when cfg.hist is on."""
    feat, cfg, published, corrected = prepare(hosp, last, as_of, sig, lags, pop, cfg, states_only,
                                              return_corrected=True)
    out = forecast_from_features(feat, last, cfg, hist if cfg.hist else None)
    # the other model families get the hospital data as published (exactly how they were tested)
    keep = (lambda d: d[~d["location"].isin(["US", "11", "72"])]) if states_only else (lambda d: d)
    g_in = keep(published)
    if cfg.gbq_weight or "gbq" in cfg.families:
        from models.gbq import GBQ, gbq_forecast
        t_in = keep(corrected[corrected["date"] <= last]) if cfg.trees_corrected else g_in
        if cfg.gbq_ed and (not len(sig.get("ed", [])) or sig["ed"]["date"].max() < last):
            # the trees were tested with the newest week's ER visits; without them, use v3.5's trees
            print(f"WARNING: ER visits not published for {last.date()}: trees run without ER visits (v3.5 trees)")
            cfg = replace(cfg, gbq_ed=False)
        g = gbq_forecast(t_in, last, pop, replace(GBQ, bags=cfg.gbq_bags, spatial=cfg.gbq_spatial,
                                                 seasonpos=cfg.gbq_context, national=cfg.gbq_context, ed=cfg.gbq_ed),
                         ed=sig.get("ed") if cfg.gbq_ed else None)
    if cfg.families:
        # v3: equal-weight quantile average of the families (no weight to tune, see V3_PROTOCOL.md)
        comps = [out]
        if "gbq" in cfg.families:
            comps.append(g)
        if "qar" in cfg.families:
            from models.qar import qar_forecast
            qa = {k: v for k, v in qar_forecast(g_in, last, pop).items() if k[0] not in cfg.qar_skip}
            if cfg.qar_tail != (1.0, 1.0):
                qa = {k: qar_tail(v, *cfg.qar_tail) for k, v in qa.items()}
            comps.append(qa)
        short = sorted({key[0] for key in out if any(key not in c for c in comps)} - set(cfg.qar_skip))
        if short:
            # a family can't forecast a place (e.g. a missing recent week): the others carry it
            import sys
            print(f"NOTE: not every model family forecast {', '.join(short)}; averaged the ones that did",
                  file=sys.stderr)
        out = {key: np.maximum.accumulate(np.maximum(np.mean([c[key] for c in comps if key in c], axis=0), 0))
               for key in out}
    elif cfg.gbq_weight:
        # v2: 6. blend with the tree model trained on ~10 seasons of history (models/gbq.py)
        a = cfg.gbq_weight
        for key in out:
            if key in g:
                out[key] = np.maximum.accumulate(np.maximum((1 - a) * out[key] + a * g[key], 0))
    return out
