"""Experiments on the PRODUCTION engine (models/engine.py): each variant is an engine.Config, so a
winner needs no porting. Decisions on DEV seasons only (2023-24, 2024-25).

  python -m backtest.exp2 [--scope states|all] name1 name2 ...

scope=states scores the 50 states (what the CDC comparison uses); scope=all also scores the US
total, DC and Puerto Rico (what the CDC's official evaluation includes).
"""
from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import pandas as pd

from backtest.harness import DEV, Replay, compare, verdict
from backtest.run_backtest import final_hosp
from backtest.run_v2 import EdArchive, ROOT
from backtest.wis import QUANTILES
from models import engine
from models.models import K_STEPS, baseline

BASE = engine.LINEAR
VARIANTS = {
    "loc_scale_05": replace(BASE, loc_scale=0.5),
    "loc_scale_10": replace(BASE, loc_scale=1.0),
    "hist": replace(BASE, hist=True),
}


def forecaster(rep: Replay, cfg: engine.Config, scope: str):
    ed = EdArchive(ROOT.parent / "FluSight-forecast-hub", rep.sig["ed"])
    lags = engine.lag_table(rep.vint.t)
    pop = pd.read_csv(ROOT / "data/clean/locations.csv", dtype={"location": str}).set_index("location")["population"]
    hist = engine.load_history(cfg) if cfg.hist else None

    def f(w):
        hosp = w.v if scope == "all" else w.v[~w.v["location"].isin(["US", "11", "72"])]
        sig = {"ww": rep.sig["ww"], "weather": rep.sig["weather"], "ed": ed.as_published(w.as_of, w.last)}
        return engine.forecast(hosp, w.last, w.as_of, sig, lags, pop, cfg,
                               states_only=(scope == "states"), hist=hist)
    return f


def base_forecaster(scope: str):
    def f(w):
        v = w.v if scope == "all" else w.v[~w.v["location"].isin(["US", "11", "72"])]
        out = {}
        for loc, s in v[v["date"] <= w.last].groupby("location"):
            s = s.set_index("date")["value"].dropna()
            if len(s) >= 30:
                for k in K_STEPS:
                    out[(loc, k)] = baseline(s, k)
        return out
    return f


def score(rep: Replay, fc, name: str, truth: pd.Series) -> pd.DataFrame:
    rows = []
    for w in rep.weeks:
        for (loc, k), q in fc(w).items():
            y = truth.get((loc, w.last + pd.Timedelta(weeks=k)))
            if y is None or np.isnan(y):
                continue
            rows.append((w.season, w.ref, loc, k - 1, y, np.asarray(q, float)))
    df = pd.DataFrame(rows, columns=["season", "reference_date", "location", "horizon", "observed", "q"])
    Q = np.vstack(df["q"].values)
    y = df["observed"].values[:, None]
    df["wis"] = 2.0 * (((y <= Q).astype(float) - QUANTILES) * (Q - y)).mean(axis=1)
    lo, hi = Q[:, QUANTILES == 0.025][:, 0], Q[:, QUANTILES == 0.975][:, 0]
    df["cov95"] = (lo <= df["observed"]) & (df["observed"] <= hi)
    lo, hi = Q[:, QUANTILES == 0.25][:, 0], Q[:, QUANTILES == 0.75][:, 0]
    df["cov50"] = (lo <= df["observed"]) & (df["observed"] <= hi)
    df["model"] = name
    return df.drop(columns="q")


def rel_on(a: pd.DataFrame, b: pd.DataFrame, mask=None) -> float:
    key = ["season", "reference_date", "location", "horizon"]
    m = a[key + ["wis"]].merge(b[key + ["wis"]], on=key, suffixes=("_a", "_b"))
    if mask is not None:
        m = m[mask(m)]
    return m["wis_b"].sum() / m["wis_a"].sum()


def run(names, scope="states", seasons=DEV, base=BASE):
    rep = Replay(seasons)
    fh = final_hosp()
    truth = fh.set_index(["location", "date"])["value"]
    b = score(rep, base_forecaster(scope), "baseline", truth)
    cur = score(rep, forecaster(rep, base, scope), "v2", truth)
    us = (lambda m: m["location"] == "US")
    print(f"{len(rep.weeks)} weeks {seasons} scope={scope}: v2 rel {rel_on(b, cur):.4f}"
          + (f" | US-only rel {rel_on(b, cur, us):.4f}" if scope == "all" else "")
          + f" | cov50 {cur['cov50'].mean():.3f} cov95 {cur['cov95'].mean():.3f}", flush=True)
    for n in names:
        s = score(rep, forecaster(rep, VARIANTS[n], scope), n, truth)
        pt, lo, hi = compare(cur, s)
        extra = ""
        if scope == "all":
            extra = f" | US-only rel {rel_on(b, s, us):.4f}"
        print(f"{n:<16} rel {rel_on(b, s):.4f} vs v2 {pt:+6.2f}% (95% CI {lo:+.2f} to {hi:+.2f}) "
              f"{verdict(lo, hi)}{extra} | cov50 {s['cov50'].mean():.3f} cov95 {s['cov95'].mean():.3f}",
              flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["states", "all"], default="states")
    ap.add_argument("names", nargs="*")
    a = ap.parse_args()
    run(a.names or list(VARIANTS), a.scope)
