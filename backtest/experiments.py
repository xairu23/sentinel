"""Configurable forecaster for experiments. Every idea is a switch so it can be tested ALONE against
the current design, on the DEV seasons only (see backtest/harness.py).

Run:  python -m backtest.experiments <experiment-name> [...]
"""
from __future__ import annotations

import itertools
import sys
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import statsmodels.api as sm

from backtest.harness import DEV, HOLDOUT, Replay, compare, rel, verdict
from backtest.wis import QUANTILES
from models.models import BASE_COLS, K_STEPS, make_features
from models.seasons import season_of

POP = None  # filled lazily: location -> population


def _pop():
    global POP
    if POP is None:
        loc = pd.read_csv("data/clean/locations.csv", dtype={"location": str})
        POP = loc.set_index("location")["population"]
    return POP


@dataclass(frozen=True)
class Cfg:
    signals: tuple = ("ed", "ww", "weather")   # every combination of these is an ensemble member
    base_cols: tuple = tuple(BASE_COLS)
    extra_cols: tuple = ()                      # extra engineered features (see add_extra)
    train_weeks: int = 156
    vol_scale: bool = True
    bias_shrink: float = 0.3
    bias_window: int = 6
    backfill: int = 0                           # correct the newest N published weeks (0 = off)
    backfill_m: int = 10                        # how many matured past weeks to learn ratios from
    backfill_shrink: float = 1.0
    weights: str = "none"                       # "none" | "sqrt" | "log" -- align fit with count-scale WIS
    spread: float = 1.0                         # multiply interval width (calibration)
    spread_by_h: tuple = ()                     # per-horizon spread multipliers (k=1..4)
    combine: str = "mean"                       # "mean" | "median" of member quantiles
    qr: bool = False                            # linear quantile regression instead of OLS+residuals
    resid: str = "insample"                     # "insample" | "recent" residual window for intervals
    hist: str = ""                              # "" | "member" (extra ensemble member) | "all" (every member)
    hist_sources: tuple = ("flusurv", "iliplus")
    hist_weight: float = 1.0                    # weight of each historical row relative to a recent one
    hist_min: float = 0.0                       # drop historical rows below this level (noisy at tiny values)
    resid_bins: int = 0                         # >0: interval width from errors of similar-sized weeks only
    ed_delay: int = 1                           # 0 = use ER data for the same newest week as hospital data
    ed_haircut: float = 1.028                   # newest ER week's typical upward revision (measured)


def add_extra(p: pd.DataFrame, extra: tuple) -> pd.DataFrame:
    """Extra features, all computed from data up to each row's own date (no future)."""
    if not extra:
        return p
    p = p.copy()
    pop = p["location"].map(_pop())
    if "cum" in extra or "cumrel" in extra:
        # cumulative admissions so far this season (season starts in week 31, early August),
        # per 100k people: as more people have already been infected, fewer are left to infect,
        # so growth slows -- the main reason an epidemic peaks.
        season = pd.Series(season_of(p["date"]), index=p.index)
        p["_season"] = season.values
        cs = p.groupby(["location", "_season"])["value"].cumsum()
        p["cum"] = np.log1p(cs / pop * 1e5)
        if "cumrel" in extra:
            # relative to this state's own previous full seasons (past seasons only)
            tot = p.groupby(["location", "_season"])["value"].sum().rename("tot").reset_index()
            tot["prev_mean"] = tot.groupby("location")["tot"].transform(
                lambda s: s.shift(1).expanding().mean())
            p = p.merge(tot[["location", "_season", "prev_mean"]], on=["location", "_season"], how="left")
            p["cumrel"] = np.log1p(cs.values) - np.log1p(p["prev_mean"])
            p = p.drop(columns="prev_mean")
        p = p.drop(columns="_season")
    if "lpc" in extra:
        p["lpc"] = np.log1p(p["value"] / pop * 1e5 * 10)  # current level per capita
    if "cum_d1" in extra:
        p["cum_d1"] = p["cum"] * p["d1"]
    if "sm1" in extra:
        p["sm1"] = p.groupby("location")["ly"].diff(2) / 2  # 2-week average growth (less noisy)
    if any(c.startswith("nat") for c in extra):
        # national growth from the sum of the states (same published version, same week): the
        # national curve is far less noisy than any one state's, and states tend to follow it
        nat = (p[~p["location"].isin(["US"])].groupby("date")["value"].sum().sort_index())
        nly = np.log1p(nat)
        n1 = nly.diff(1)
        n2 = nly.diff(2) - n1
        p["nat1"] = p["date"].map(n1)
        p["nat2"] = p["date"].map(n2)
    return p


HIST = {}  # source -> feature table (built once)


def load_history(sources: tuple, extra: tuple) -> pd.DataFrame:
    """Long history of flu activity from BEFORE July 2022 (more than a year before the first
    forecast week ever tested), turned into the same base features as the hospital data.
    flusurv = CDC FluSurv-NET lab-confirmed flu hospitalization rate per 100k (2003-2022)
    iliplus = outpatient influenza-like-illness % x lab flu positivity (2015-2022), every state
    Both are rescaled to state-sized pseudo-counts so log1p behaves like it does on admissions."""
    key = (sources, extra)
    if key in HIST:
        return HIST[key]
    from scripts.lab_clean import epiweek_to_saturday
    h = pd.read_csv("data/raw/sentinel_history.csv")
    h["date"] = epiweek_to_saturday(h["epiweek"])
    h = h[h["date"] < "2022-07-01"]
    loc = pd.read_csv("data/clean/locations.csv", dtype={"location": str})
    pop_by_abbr = loc.set_index(loc["abbreviation"].str.upper())["population"]
    out = []
    if "flusurv" in sources:
        f = h[h["source"] == "flusurv"].copy()
        ab = f["location"].str.split("_").str[0].str.upper().replace({"NETWORK": "US"})
        pop = ab.map(pop_by_abbr)
        f["value"] = f["v1"] * pop / 1e5
        f["id"] = "fs_" + f["location"]
        f["rate"] = f["v1"]
        out.append(f[["id", "date", "value", "rate"]])
    if "iliplus" in sources:
        g = h[h["source"] == "iliplus"].copy()
        pop = g["location"].str.upper().map(pop_by_abbr)
        g["rate"] = g["v1"] * g["v2"] / 100 * 4  # roughly admissions-per-100k scale at the peak
        g["value"] = g["rate"] * pop / 1e5
        g["id"] = "ip_" + g["location"]
        out.append(g[["id", "date", "value", "rate"]])
    d = pd.concat(out).dropna()
    rows = []
    for i, s in d.groupby("id"):
        s = s.set_index("date").sort_index()
        s = s[~s.index.duplicated()]
        grid = pd.date_range(s.index.min(), s.index.max(), freq="7D")
        s = s.reindex(grid)  # gaps (summers) stay empty, so no change is computed across them
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
    if "cum" in extra:
        season = pd.Series(season_of(p["date"]), index=p.index)
        p["cum"] = np.log1p(p.assign(_s=season.values).groupby(["location", "_s"])["rate"]
                            .cumsum().fillna(0))
        if "cum_d1" in extra:
            p["cum_d1"] = p["cum"] * p["d1"]
    HIST[key] = p
    return p


def hist_training(cfg: Cfg, cols: list, k: int) -> pd.DataFrame:
    p = load_history(cfg.hist_sources, cfg.extra_cols)
    tgt = p.groupby("location")["ly"].shift(-k) - p["ly"]
    t = p.assign(target=tgt)
    base = [c for c in cols if c in t.columns]
    t = t.replace([np.inf, -np.inf], np.nan).dropna(subset=base + ["target"])
    if cfg.hist_min:
        t = t[t["rate"] >= cfg.hist_min]
    return t


def backfill_correct(rep: Replay, w, cfg: Cfg) -> pd.DataFrame:
    """Scale up the newest `cfg.backfill` published weeks by how much such weeks were revised in the
    past. Learned only from versions published BEFORE this week, compared with this week's version
    of weeks that are now >= 4 weeks old (essentially final by then)."""
    key = ("lag_tables",)
    if key not in rep.__dict__:
        t = rep.vint.t
        maxd = t.groupby("as_of")["date"].max().rename("maxd")
        tt = t.merge(maxd, left_on="as_of", right_index=True)
        tt["lag"] = (tt["maxd"] - tt["date"]).dt.days // 7
        first = (tt[tt["lag"] <= 2].sort_values("as_of")
                 .groupby(["location", "date", "lag"], as_index=False).first())
        rep.__dict__[key] = first[["location", "date", "lag", "as_of", "value"]]
    first = rep.__dict__[key]
    panel = w.panel.copy()
    cur = panel.set_index(["location", "date"])["value"]
    for lag in range(cfg.backfill):
        f = first[(first["lag"] == lag) & (first["as_of"] < w.as_of)
                  & (first["date"] <= w.last - pd.Timedelta(weeks=4))]
        recent_dates = np.sort(f["date"].unique())[-cfg.backfill_m:]
        f = f[f["date"].isin(recent_dates)]
        c = cur.reindex(pd.MultiIndex.from_frame(f[["location", "date"]])).values
        ok = (f["value"].values > 0) & (c >= 5)
        lr = pd.Series(np.log(c[ok] / f["value"].values[ok]), index=f["location"].values[ok])
        pooled = lr.median() if len(lr) else 0.0
        per = lr.groupby(level=0).median()
        n = lr.groupby(level=0).size()
        # shrink each state's own ratio toward the national one when it has few observations
        a = n / (n + 5)
        est = (a * per + (1 - a) * pooled).clip(-0.2, 0.5) * cfg.backfill_shrink
        tgt = w.last - pd.Timedelta(weeks=lag)
        m = panel["date"] == tgt
        panel.loc[m, "value"] = panel.loc[m, "value"] * np.exp(
            panel.loc[m, "location"].map(est).fillna(pooled * cfg.backfill_shrink).values)
    return panel


def fit_one(feat: pd.DataFrame, last, cols: list, cfg: Cfg, use_hist: bool = False) -> dict:
    if use_hist and not cfg.qr:
        return fit_one_hist(feat, last, cols, cfg)
    out = {}
    p = feat
    for k in K_STEPS:
        tgt = p.groupby("location")["ly"].shift(-k) - p["ly"]
        tdate = p["date"] + pd.Timedelta(weeks=k)
        sel = (tdate <= last) & (p["date"] > last - pd.Timedelta(weeks=cfg.train_weeks))
        keep = list(dict.fromkeys(cols + ["location", "date", "value", "d1"]))  # no duplicate columns
        train = p.loc[sel, keep].assign(target=tgt[sel])
        train = train.replace([np.inf, -np.inf], np.nan).dropna(subset=cols + ["target"])
        if len(train) < 200:
            continue
        now = p[p["date"] == last].replace([np.inf, -np.inf], np.nan).dropna(subset=cols)
        if now.empty:
            continue
        X = sm.add_constant(train[cols].values, has_constant="add")
        Xn = sm.add_constant(now[cols].values, has_constant="add")
        if cfg.weights == "sqrt":
            wts = np.sqrt(train["value"].values + 1)
        elif cfg.weights == "log":
            wts = np.log1p(train["value"].values) + 1
        elif cfg.weights.startswith("recency"):
            # recent weeks count more: halflife in weeks, e.g. "recency26"
            hl = float(cfg.weights[len("recency"):])
            age = (last - train["date"]).dt.days.values / 7.0
            wts = 0.5 ** (age / hl)
        else:
            wts = None
        if cfg.qr:
            taus = np.array([0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975])
            coarse = np.column_stack([sm.QuantReg(train["target"].values, X).fit(q=t, max_iter=2000).predict(Xn)
                                      for t in taus])
            coarse = np.sort(coarse, axis=1)
            for i, (loc, ly0) in enumerate(now[["location", "ly"]].itertuples(index=False)):
                fine = np.interp(QUANTILES, taus, coarse[i])
                med = fine[QUANTILES == 0.5][0]
                fine = med + (fine - med) * cfg.spread
                q = np.expm1(ly0 + np.clip(fine, -2, 2))
                out[(loc, k)] = np.maximum.accumulate(np.maximum(q, 0))
            continue
        model = sm.WLS(train["target"].values, X, weights=wts).fit() if wts is not None else \
            sm.OLS(train["target"].values, X).fit()
        resid = train["target"].values - model.predict(X)
        if cfg.resid == "recent":
            r_sel = train["date"].values > np.datetime64(last - pd.Timedelta(weeks=52))
            resid_for_q = resid[r_sel] if r_sel.sum() > 100 else resid
        else:
            resid_for_q = resid
        resid_q = np.quantile(resid_for_q, QUANTILES)
        med_r = np.median(resid_for_q)
        if cfg.resid_bins:
            # error spread by size: bins of the log level at the time of each training row
            lyt = np.log1p(train["value"].values)
            edges = np.quantile(lyt, np.linspace(0, 1, cfg.resid_bins + 1)[1:-1])
            bin_t = np.searchsorted(edges, lyt)
            bin_q = {b: (np.quantile(resid[bin_t == b], QUANTILES), np.median(resid[bin_t == b]))
                     for b in range(cfg.resid_bins) if (bin_t == b).sum() > 50}
        mult = cfg.spread * (cfg.spread_by_h[k - 1] if cfg.spread_by_h else 1.0)
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
            if cfg.resid_bins:
                b = int(np.searchsorted(edges, ly0))
                if b in bin_q:
                    rq, mr = bin_q[b]
            sr = mr + (rq - mr) * vr * mult
            b = 0.0
            if cfg.bias_shrink:
                b = per_b.get(loc, glob_b)
                b = np.clip((b if pd.notna(b) else 0.0) * cfg.bias_shrink, -1, 1)
            q = np.expm1(ly0 + np.clip(m + b + sr, -2.0, 2.0))
            out[(loc, k)] = np.maximum.accumulate(np.maximum(q, 0))
    return out


def fit_one_hist(feat: pd.DataFrame, last, cols: list, cfg: Cfg) -> dict:
    """OLS trained on recent data PLUS the long pre-2022 history. Columns the history doesn't have
    (wastewater, ER, weather) are set to 0 there, with an `is_hist` indicator, so their effects are
    learned only from recent data while the core growth dynamics learn from ~20 seasons."""
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
        H = hist_training(cfg, cols, k)
        Hx = pd.DataFrame({c: (H[c].values if c in H.columns else 0.0) for c in cols}, index=H.index)
        Xc = np.column_stack([train[cols].values, np.zeros(len(train))])
        Xh = np.column_stack([Hx.values, np.ones(len(Hx))])
        X = sm.add_constant(np.vstack([Xc, Xh]), has_constant="add")
        y = np.concatenate([train["target"].values, H["target"].values])
        if cfg.weights.startswith("recency"):
            # recent rows keep the same recency weighting as without history; historical rows
            # (2010-2020) get a fixed weight, otherwise recency would wipe them out entirely
            hl = float(cfg.weights[len("recency"):])
            w_cur = 0.5 ** ((last - train["date"]).dt.days.values / 7.0 / hl)
        else:
            w_cur = np.ones(len(train))
        wts = np.concatenate([w_cur, np.full(len(Hx), cfg.hist_weight)])
        model = sm.WLS(y, X, weights=wts).fit()
        Xc1 = sm.add_constant(Xc, has_constant="add")
        resid = train["target"].values - model.predict(Xc1)  # interval width from RECENT errors only
        resid_q = np.quantile(resid, QUANTILES)
        med_r = np.median(resid)
        mult = cfg.spread * (cfg.spread_by_h[k - 1] if cfg.spread_by_h else 1.0)
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
            sr = med_r + (resid_q - med_r) * vr * mult
            b = 0.0
            if cfg.bias_shrink:
                b = per_b.get(loc, glob_b)
                b = np.clip((b if pd.notna(b) else 0.0) * cfg.bias_shrink, -1, 1)
            q = np.expm1(ly0 + np.clip(m + b + sr, -2.0, 2.0))
            out[(loc, k)] = np.maximum.accumulate(np.maximum(q, 0))
    return out


def members(cfg: Cfg) -> list[tuple]:
    return [c for r in range(len(cfg.signals) + 1) for c in itertools.combinations(cfg.signals, r)]


def cols_for(combo: tuple, cfg: Cfg) -> list:
    cols = list(cfg.base_cols) + list(cfg.extra_cols)
    for s in combo:
        cols += [f"{s}1", f"{s}2", f"{s}l"]
    return cols


ED_VINT = {}


def panel_ed0(rep: Replay, w, cfg: Cfg, hosp: pd.DataFrame) -> pd.DataFrame:
    """Attach signals with ER visits for the SAME newest week as the hospital data (that's how CDC
    publishes them: same day, same newest week). Where CDC's archived ER versions exist (Dec 2025 on)
    use exactly what was published that day; before that, use the final numbers but scale the newest
    week DOWN by its measured typical later revision (~2.8%), so it looks like a first publication."""
    import backtest.run_backtest as rb
    if not ED_VINT:
        t = pd.read_csv(rb.ROOT.parent / "FluSight-forecast-hub/target-data/time-series.csv",
                        dtype={"location": str}, parse_dates=["as_of", "target_end_date"])
        t = t[t["target"] == "wk inc flu prop ed visits"]
        ED_VINT["t"] = t.rename(columns={"target_end_date": "date", "observation": "value"})
    t = ED_VINT["t"]
    vin = t[t["as_of"] == w.as_of]
    e = rep.sig["ed"].copy()
    e = e[e["date"] <= w.last]
    if len(vin):
        ve = vin[["date", "location", "value"]].copy()
        ve["ed_level"] = np.log(ve["value"] + 1e-4)
        e = ve[["date", "location", "ed_level"]]
    else:
        newest = e["date"] == w.last
        e.loc[newest, "ed_level"] = e.loc[newest, "ed_level"] - np.log(cfg.ed_haircut)
    sig = dict(rep.sig, ed=e)
    old = rb.ED_DELAY
    rb.ED_DELAY = 0
    try:
        return rb.attach_signals(hosp, sig)
    finally:
        rb.ED_DELAY = old


def make_forecaster(rep: Replay, cfg: Cfg):
    def f(w):
        if cfg.ed_delay == 0:
            hosp = w.v
            if cfg.backfill:
                hosp = backfill_correct(rep, w, cfg)[["date", "location", "value"]]
            feat = make_features(panel_ed0(rep, w, cfg, hosp))
        elif cfg.backfill:
            feat = make_features(backfill_correct(rep, w, cfg))
        else:
            feat = w.feat
        feat = add_extra(feat, cfg.extra_cols)
        preds = [fit_one(feat, w.last, cols_for(c, cfg), cfg, use_hist=(cfg.hist == "all"))
                 for c in members(cfg)]
        if cfg.hist == "member":
            preds.append(fit_one(feat, w.last, cols_for((), cfg), cfg, use_hist=True))
        keys = set().union(*preds)
        out = {}
        for key in keys:
            qs = [p[key] for p in preds if key in p]
            q = np.median(qs, axis=0) if cfg.combine == "median" else np.mean(qs, axis=0)
            out[key] = np.maximum.accumulate(np.maximum(q, 0))
        return out
    return f


CURRENT = Cfg()

EXPERIMENTS = {
    "no_bias": replace(CURRENT, bias_shrink=0.0),
    "backfill1": replace(CURRENT, backfill=1),
    "backfill2": replace(CURRENT, backfill=2),
    "backfill1_half": replace(CURRENT, backfill=1, backfill_shrink=0.5),
    "wls_sqrt": replace(CURRENT, weights="sqrt"),
    "wls_log": replace(CURRENT, weights="log"),
    "cum": replace(CURRENT, extra_cols=("cum",)),
    "cumrel": replace(CURRENT, extra_cols=("cumrel",)),
    "cum_d1": replace(CURRENT, extra_cols=("cum", "cum_d1")),
    "lpc": replace(CURRENT, extra_cols=("lpc",)),
    "sm1": replace(CURRENT, extra_cols=("sm1",)),
    "median_combine": replace(CURRENT, combine="median"),
    "spread_1.2": replace(CURRENT, spread=1.2),
    "spread_1.4": replace(CURRENT, spread=1.4),
    "spread_0.9": replace(CURRENT, spread=0.9),
    "resid_recent": replace(CURRENT, resid="recent"),
    "no_vol": replace(CURRENT, vol_scale=False),
    "recency26": replace(CURRENT, weights="recency26"),
    "recency52": replace(CURRENT, weights="recency52"),
    "recency104": replace(CURRENT, weights="recency104"),
    "nat": replace(CURRENT, extra_cols=("nat1", "nat2"), ),
    "qr": replace(CURRENT, qr=True, bias_shrink=0.0),
    "combo1": replace(CURRENT, ed_delay=0, weights="recency52", extra_cols=("cum", "cum_d1"), backfill=1),
    "combo1_r104": replace(CURRENT, ed_delay=0, weights="recency104", extra_cols=("cum", "cum_d1"), backfill=1),
    "rbins3": replace(CURRENT, resid_bins=3),
    "rbins5": replace(CURRENT, resid_bins=5),
    "ed0": replace(CURRENT, ed_delay=0),
    "ed0_nohaircut": replace(CURRENT, ed_delay=0, ed_haircut=1.0),
    "hist_member": replace(CURRENT, hist="member"),
    "hist_all": replace(CURRENT, hist="all"),
    "hist_all_w03": replace(CURRENT, hist="all", hist_weight=0.3),
    "hist_all_fs": replace(CURRENT, hist="all", hist_sources=("flusurv",)),
    "hist_all_ip": replace(CURRENT, hist="all", hist_sources=("iliplus",)),
    "hist_all_min": replace(CURRENT, hist="all", hist_min=0.5),
}


def run(names: list[str], seasons=DEV, base: Cfg = CURRENT):
    rep = Replay(seasons)
    b = rep.score(rep.baseline_forecaster, "baseline")
    cur = rep.score(make_forecaster(rep, base), "current")
    print(f"{len(rep.weeks)} weeks, seasons {seasons}. current rel WIS vs baseline: {rel(cur, b):.4f}", flush=True)
    res = {}
    for n in names:
        cfg = EXPERIMENTS[n] if isinstance(n, str) else n
        s = rep.score(make_forecaster(rep, cfg), str(n))
        pt, lo, hi = compare(cur, s)
        res[n] = (rel(s, b), pt, lo, hi)
        print(f"{str(n):<18} rel {rel(s, b):.4f}  vs current {pt:+6.2f}% (95% CI {lo:+.2f} to {hi:+.2f})  "
              f"{verdict(lo, hi)}   cov50 {s['cov50'].mean():.3f} cov95 {s['cov95'].mean():.3f}", flush=True)
    return res


COMBO1 = EXPERIMENTS["combo1"]
EXPERIMENTS.update({
    "c1_rbins3": replace(COMBO1, resid_bins=3),
    "c1_nobias": replace(COMBO1, bias_shrink=0.0),
    "c1_bias15": replace(COMBO1, bias_shrink=0.15),
    "c1_backfill2": replace(COMBO1, backfill=2),
    "c1_novol": replace(COMBO1, vol_scale=False),
    "c1_spread09": replace(COMBO1, spread=0.9),
    "c1_r26": replace(COMBO1, weights="recency26"),
    "c1_sp09_rb3": replace(COMBO1, spread=0.9, resid_bins=3),
    "c1_sp095": replace(COMBO1, spread=0.95),
    "c1_hist_all": replace(COMBO1, hist="all"),
    "c1_hist_member": replace(COMBO1, hist="member"),
    "c1_hist_w03": replace(COMBO1, hist="all", hist_weight=0.3),
    "c1_hist_fs": replace(COMBO1, hist="all", hist_sources=("flusurv",), hist_min=0.5, hist_weight=0.3),
    "c1_hist_ip": replace(COMBO1, hist="all", hist_sources=("iliplus",), hist_min=0.5, hist_weight=0.3),
    "c1_hist_min": replace(COMBO1, hist="all", hist_min=0.5),
    "c1_hist_w03_min": replace(COMBO1, hist="all", hist_min=0.5, hist_weight=0.3),
    "c1_hist_w01_min": replace(COMBO1, hist="all", hist_min=0.5, hist_weight=0.1),
    "c1_hist_member_min": replace(COMBO1, hist="member", hist_min=0.5, hist_weight=0.3),
})

if __name__ == "__main__":
    args = sys.argv[1:]
    base = CURRENT
    if args and args[0].startswith("base="):
        base = EXPERIMENTS[args[0][5:]]
        args = args[1:]
    run(args or list(EXPERIMENTS), base=base)
