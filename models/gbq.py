"""Sentinel GBQ: gradient-boosted quantile trees trained on MANY seasons of pooled flu data.

After the UMass Reich Lab "flusion" GBQ model (github.com/reichlab/flusion, MIT licence): instead of
learning from Sentinel's ~3 seasons of hospital data alone, the trees also learn the SHAPE of flu
seasons (rises, peaks, turns) from ~10 seasons of FluSurv-NET hospitalisation rates and ILI+ by
state (all from before July 2022). Every series is put on a common scale so shapes transfer:
  rate per 100k -> fourth root -> divide by the series' own 95th percentile (in season)
  -> centre on its own in-season mean.
Features describe the recent trajectory (local polynomial fits over the last 3-6 weeks: level,
slope, curvature; rolling means; their values 1-2 weeks earlier), time of season and population.
The target is the change on that scale 1-4 weeks ahead; one model per quantile level, bagged over
seasons, median across bags.

Leak safety: the hospital series' scale/centre use only the data published up to the forecast week
(the panel passed in), and all history ends before July 2022.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from backtest.wis import QUANTILES
from models.models import K_STEPS
from models.seasons import season_of, season_week as _season_week

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class GbqConfig:
    bags: int = 3
    bag_frac: float = 0.7
    taus: tuple = (0.01, 0.025, 0.1, 0.25, 0.5, 0.75, 0.9, 0.975, 0.99)
    n_estimators: int = 100
    num_leaves: int = 31
    learning_rate: float = 0.1
    use_history: bool = True
    sources: tuple = ("flusurv", "iliplus")
    spatial: bool = False          # neighbours' level and slope as features (round 4, idea 18)
    xmas: bool = False             # weeks from Christmas as a feature (idea 6)
    nhsn_weight: float = 1.0       # weight of hospital (NHSN) rows vs history rows (idea 12, trees part)
    seasonpos: bool = False        # running season maximum, distance below it, weeks since it (round 31)
    national: bool = False         # average level / slope of all places of the same source, and own level vs it (round 31)
    ed: bool = False               # ER flu share level / slopes for hospital rows (round 32); needs `ed` passed in


GBQ = GbqConfig()


def season_week(d: pd.Series) -> np.ndarray:
    return _season_week(d)  # models/seasons.py: ISO-year based, no week-53 glitch


def season_id(d: pd.Series) -> np.ndarray:
    return season_of(d)


def _taylor(y: np.ndarray, window: int, degree: int) -> np.ndarray:
    """Least-squares polynomial fit over each trailing window; coefficients at the newest point
    (c0 = level, c1 = slope, c2 = curvature). NaN if the window has a gap."""
    t = np.arange(-(window - 1), 1, dtype=float)
    P = np.linalg.pinv(np.vander(t, degree + 1, increasing=True))  # (degree+1, window)
    out = np.full((len(y), degree + 1), np.nan)
    if len(y) >= window:
        win = sliding_window_view(y, window)  # (n-window+1, window)
        out[window - 1:] = win @ P.T
    return out


def _series_features(cs: np.ndarray) -> dict:
    f = {"cs": cs}
    for w, d in ((4, 2), (6, 2), (3, 1), (5, 1)):
        c = _taylor(cs, w, d)
        for j in range(d + 1):
            f[f"t{w}d{d}c{j}"] = c[:, j]
    s = pd.Series(cs)
    f["rm2"] = s.rolling(2).mean().values
    f["rm4"] = s.rolling(4).mean().values
    base = list(f)
    for lag in (1, 2):
        for k in base:
            f[f"{k}_l{lag}"] = np.concatenate([np.full(lag, np.nan), f[k][:-lag]])
    return f


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """df: series_id, source, loc_key, date, rate (per 100k, source-adjusted), pop -> adds cs and the
    per-series factors needed to invert it."""
    df = df.copy()
    df["it"] = (df["rate"] + 0.01) ** 0.25
    df["sw"] = season_week(df["date"])
    ins = df["it"].where(df["sw"].between(10, 45))
    df["scale"] = ins.groupby(df["series_id"]).transform(lambda x: x.quantile(0.95))
    df["cs"] = df["it"] / (df["scale"] + 0.01)
    ins_cs = df["cs"].where(df["sw"].between(10, 45))
    df["center"] = ins_cs.groupby(df["series_id"]).transform("mean")
    df["cs"] = df["cs"] - df["center"]
    return df


def featurize(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for sid, g in df.groupby("series_id", sort=False):
        g = g.sort_values("date").drop_duplicates("date", keep="last")
        grid = pd.date_range(g["date"].min(), g["date"].max(), freq="7D")
        g = g.set_index("date").reindex(grid)
        g.index.name = "date"
        g["series_id"] = sid
        for c in ("source", "loc_key", "pop", "scale", "center"):
            g[c] = g[c].ffill().bfill()
        g = g.reset_index()
        g["sw"] = season_week(g["date"])
        g["season"] = season_id(g["date"])
        for k, v in _series_features(g["cs"].values).items():
            g[k] = v
        for h in K_STEPS:
            g[f"tgt{h}"] = g["cs"].shift(-h) - g["cs"]
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    out["log_pop"] = np.log(out["pop"])
    return out


FEATS = None  # filled on first featurize


def feature_names(df: pd.DataFrame) -> list:
    skip = {"series_id", "source", "loc_key", "date", "rate", "pop", "it", "scale", "center", "season",
            "value"} | {f"tgt{h}" for h in K_STEPS}
    return [c for c in df.columns if c not in skip] + ["src", "loc_cat", "horizon"]


_HIST_CACHE = {}


def history_frame(cfg: GbqConfig, pop_by_abbr: pd.Series, abbr_to_fips: dict) -> pd.DataFrame:
    key = cfg.sources
    if key in _HIST_CACHE:
        return _HIST_CACHE[key]
    from scripts.lab_clean import epiweek_to_saturday
    h = pd.read_csv(ROOT / "data/history/sentinel_history.csv")
    h["date"] = epiweek_to_saturday(h["epiweek"])
    h = h[h["date"] < "2022-07-01"]
    parts = []
    if "flusurv" in cfg.sources:
        f = h[h["source"] == "flusurv"].copy()
        ab = f["location"].str.split("_").str[0].str.upper()
        parts.append(pd.DataFrame({"series_id": "fs_" + f["location"], "source": "flusurv",
                                   "loc_key": ab.map(abbr_to_fips), "date": f["date"],
                                   "rate": (f["v1"] + np.exp(-3)) / 2.5, "pop": ab.map(pop_by_abbr)}))
    if "iliplus" in cfg.sources:
        g = h[h["source"] == "iliplus"].copy()
        ab = g["location"].str.upper()
        parts.append(pd.DataFrame({"series_id": "ip_" + g["location"], "source": "iliplus",
                                   "loc_key": ab.map(abbr_to_fips), "date": g["date"],
                                   "rate": (g["v1"] * g["v2"] / 100 + np.exp(-7)) * 4,
                                   "pop": ab.map(pop_by_abbr)}))
    d = pd.concat(parts).dropna(subset=["rate", "pop"])
    feat = featurize(transform(d))
    _HIST_CACHE[key] = feat
    return feat


def stack_horizons(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for h in K_STEPS:
        r = df.copy()
        r["horizon"] = h
        r["target"] = r[f"tgt{h}"]
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


SRC_CODE = {"nhsn": 0, "flusurv": 1, "iliplus": 2}


def add_neighbours(allf: pd.DataFrame, locs: pd.DataFrame) -> pd.DataFrame:
    """Average level (cs) and 4-week slope of the bordering states, same source and same week."""
    from models.neighbors import neighbor_pairs
    pairs = pd.DataFrame(neighbor_pairs(locs), columns=["loc_key", "nb"])
    v = (allf.dropna(subset=["loc_key"]).groupby(["source", "date", "loc_key"], as_index=False)
         [["cs", "t4d2c1"]].mean().rename(columns={"loc_key": "nb", "cs": "nb_cs", "t4d2c1": "nb_slope"}))
    m = pairs.merge(v, on="nb").groupby(["source", "date", "loc_key"], as_index=False)[["nb_cs", "nb_slope"]].mean()
    return allf.merge(m, on=["source", "date", "loc_key"], how="left")


def add_season_position(allf: pd.DataFrame) -> pd.DataFrame:
    """Per series and season, using only weeks up to each row: running maximum of cs, distance below it,
    and weeks since it was set (round 31, after DMAPRIME-HYB / Schnitzels / GT_Influencers / UMass-peakGB)."""
    allf = allf.sort_values(["series_id", "date"]).reset_index(drop=True)
    keys = [allf["series_id"], allf["season"]]
    allf["runmax"] = allf.groupby(keys, sort=False)["cs"].cummax()
    allf["below_max"] = allf["cs"] - allf["runmax"]
    pos = pd.Series(np.arange(len(allf), dtype=float))
    at_max = pos.where(allf["cs"] >= allf["runmax"] - 1e-12)
    last_max = at_max.groupby(keys, sort=False).ffill()
    allf["wks_since_max"] = (pos - last_max).values
    return allf


def add_national(allf: pd.DataFrame) -> pd.DataFrame:
    """Average level (cs) and 4-week slope over all places of the same source on the same week, and each
    place's level relative to it (round 31). Uses only that week's values, which are known at that week."""
    v = (allf.groupby(["source", "date"], as_index=False)
         .agg(nat_cs=("cs", "mean"), nat_slope=("t4d2c1", "mean")))
    allf = allf.merge(v, on=["source", "date"], how="left")
    allf["rel_nat"] = allf["cs"] - allf["nat_cs"]
    return allf


def add_ed(allf: pd.DataFrame, ed: pd.DataFrame) -> pd.DataFrame:
    """Round 32: ER flu share features for hospital (nhsn) rows; other sources stay missing.
    ed: date, location, ed_level (log ER flu share), as published at the forecast date."""
    e = ed.dropna(subset=["ed_level"]).sort_values(["location", "date"]).drop_duplicates(["location", "date"], keep="last")
    parts = []
    for loc, g in e.groupby("location", sort=False):
        grid = pd.date_range(g["date"].min(), g["date"].max(), freq="7D")
        y = g.set_index("date")["ed_level"].reindex(grid).values
        sw = season_week(pd.Series(grid))
        ins = np.where((sw >= 10) & (sw <= 45), y, np.nan)
        f = pd.DataFrame({"date": grid, "loc_key": loc, "ed_lvl": y,
                          "ed_z": y - np.nanmean(ins) if np.isfinite(ins).any() else np.nan,
                          "ed_s3": _taylor(y, 3, 1)[:, 1], "ed_s5": _taylor(y, 5, 1)[:, 1],
                          "ed_d1": np.concatenate([[np.nan], np.diff(y)])})
        parts.append(f)
    f = pd.concat(parts, ignore_index=True)
    f["source"] = "nhsn"
    allf = allf.merge(f, on=["source", "date", "loc_key"], how="left")
    # 2-week change of (log ER share - log admissions rate): ER rising faster than admissions is a lead
    allf = allf.sort_values(["series_id", "date"]).reset_index(drop=True)
    gap = allf["ed_lvl"] - np.log(allf["rate"].clip(lower=1e-6))
    allf["ed_gap_d2"] = gap - gap.groupby(allf["series_id"], sort=False).shift(2)
    return allf.drop(columns=["ed_lvl"])


def gbq_forecast(hosp: pd.DataFrame, last: pd.Timestamp, pop: pd.Series, cfg: GbqConfig = GBQ,
                 seed: int = 0, ed: pd.DataFrame | None = None) -> dict:
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
        frames.append(history_frame(cfg, pop_by_abbr, abbr_to_fips))
    allf = pd.concat(frames, ignore_index=True)
    allf["src"] = allf["source"].map(SRC_CODE)
    loc_codes = {k: i for i, k in enumerate(sorted(allf["loc_key"].dropna().unique()))}
    allf["loc_cat"] = allf["loc_key"].map(loc_codes)
    if cfg.spatial:
        allf = add_neighbours(allf, locs)
    if cfg.seasonpos:
        allf = add_season_position(allf)
    if cfg.national:
        allf = add_national(allf)
    if cfg.ed:
        if ed is None:
            raise ValueError("cfg.ed needs the ER data (ed=...)")
        allf = add_ed(allf, ed[ed["date"] <= last])
    if cfg.xmas:
        start = season_of(allf["date"])
        xmas_sw = {y: int(season_week(pd.Series([pd.Timestamp(f"{y}-12-25")]))[0]) for y in np.unique(start)}
        allf["dxmas"] = allf["sw"].values - pd.Series(start).map(xmas_sw).values
    feats = feature_names(allf.drop(columns=["src", "loc_cat"], errors="ignore"))

    test = allf[(allf["source"] == "nhsn") & (allf["date"] == last)].copy()
    tr = allf[allf["sw"].between(5, 45)]
    train = stack_horizons(tr)
    train = train[train["target"].notna()]
    # a target must be known by `last` (true by construction: the panel ends at `last`)
    train = train[(train["source"] != "nhsn") | (train["date"] + pd.to_timedelta(7 * train["horizon"], "D") <= last)]
    test_h = stack_horizons(test)

    rng = np.random.default_rng(seed + int(last.strftime("%Y%m%d")))
    seasons = train["season"].astype(str) + "_" + train["source"]
    useas = seasons.unique()
    taus = np.array(cfg.taus)
    preds = np.empty((len(test_h), cfg.bags, len(taus)))
    for b in range(cfg.bags):
        pick = rng.choice(useas, size=max(1, int(len(useas) * cfg.bag_frac)), replace=False)
        m = seasons.isin(pick).values
        X, y = train.loc[m, feats], train.loc[m, "target"]
        sw_ = (np.where(train.loc[m, "source"].values == "nhsn", cfg.nhsn_weight, 1.0)
               if cfg.nhsn_weight != 1.0 else None)
        for j, tau in enumerate(taus):
            model = lgb.LGBMRegressor(objective="quantile", alpha=float(tau), n_estimators=cfg.n_estimators,
                                      num_leaves=cfg.num_leaves, learning_rate=cfg.learning_rate,
                                      verbosity=-1, random_state=int(rng.integers(1e8)),
                                      n_jobs=int(__import__("os").environ.get("GBQ_THREADS", "2")))
            model.fit(X, y, sample_weight=sw_, categorical_feature=["src", "loc_cat"])
            preds[:, b, j] = model.predict(test_h[feats])
    delta = np.sort(np.median(preds, axis=1), axis=1)  # (n, taus), sorted -> no crossing
    out = {}
    for i, r in enumerate(test_h.itertuples(index=False)):
        dq = np.interp(QUANTILES, taus, delta[i])
        it = (r.cs + dq + r.center) * (r.scale + 0.01)
        rate = np.maximum(it, 0) ** 4 - 0.01 - 0.75 ** 4
        val = np.maximum(rate, 0) * r.pop / 1e5
        loc = r.loc_key
        out[(loc, int(r.horizon))] = np.maximum.accumulate(val)
    return out
