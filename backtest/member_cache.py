"""Cache every regression member's forecast for every replay week (ER visits as really published),
so that experiments on HOW to combine the members and the tree model take seconds, not hours.

    python -m backtest.member_cache --seasons 2023-24 2024-25 2025-26 [--tag mem]

Writes backtest/cache/{tag}_{reference date}.pkl = {"ref", "season", "members": {combo: {(loc, k): q}}}.
The members are exactly what engine.forecast() averages (backtest/exp_v3.py checks that their
mean equals the engine's linear part before using them).
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.harness import DEV, HOLDOUT, Replay
from backtest.run_v2 import EdArchive
from models import engine

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "backtest/cache"


def week_inputs(rep: Replay, ed: EdArchive):
    def inputs(w):
        hosp = w.v[~w.v["location"].isin(["US", "11", "72"])]
        sig = {"ww": rep.sig["ww"], "weather": rep.sig["weather"], "ed": ed.as_published(w.as_of, w.last)}
        return hosp, sig
    return inputs


def produce(seasons, tag: str, cfg: engine.Config = engine.LINEAR):
    rep = Replay(tuple(seasons))
    ed = EdArchive(ROOT.parent / "FluSight-forecast-hub", rep.sig["ed"])
    lags = engine.lag_table(rep.vint.t)
    pop = pd.read_csv(ROOT / "data/clean/locations.csv", dtype={"location": str}).set_index("location")["population"]
    inputs = week_inputs(rep, ed)
    CACHE.mkdir(exist_ok=True)
    for w in rep.weeks:
        f = CACHE / f"{tag}_{w.ref.date()}.pkl"
        if f.exists():
            continue
        hosp, sig = inputs(w)
        m = engine.member_forecasts(hosp, w.last, w.as_of, sig, lags, pop, cfg, states_only=True)
        f.write_bytes(pickle.dumps({"ref": w.ref, "season": w.season, "members": m}))
        print(w.ref.date(), "done", flush=True)


def load(tag: str = "mem") -> dict:
    return {d["ref"]: d for d in (pickle.loads(f.read_bytes()) for f in sorted(CACHE.glob(f"{tag}_*.pkl")))}


def mean_members(members: dict) -> dict:
    out = {}
    for key in set().union(*members.values()):
        qs = [p[key] for p in members.values() if key in p]
        out[key] = np.maximum.accumulate(np.maximum(np.mean(qs, axis=0), 0))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", default=list(DEV + HOLDOUT))
    ap.add_argument("--tag", default="mem")
    ap.add_argument("--train-weeks", type=int, default=None, help="other training window (v3 idea 4)")
    a = ap.parse_args()
    from dataclasses import replace
    produce(a.seasons, a.tag, replace(engine.LINEAR, train_weeks=a.train_weeks) if a.train_weeks else engine.LINEAR)
