"""Does the GBQ tree model (models/gbq.py) improve Sentinel v2? Replays the given seasons, saves every
week's v2 and GBQ forecasts (backtest/cache/), then scores GBQ alone and blends with v2.

  python -m backtest.exp_gbq --seasons 2023-24 [--tag name]      (one process per season is fastest)
  python -m backtest.exp_gbq --score --seasons 2023-24 2024-25    (score saved forecasts, with CIs)
"""
from __future__ import annotations

import argparse
import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.harness import Replay, compare, verdict
from backtest.run_v2 import make_v2_forecaster
from models import engine
from models.gbq import GBQ, gbq_forecast

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "backtest/cache"
BLENDS = (0.0, 0.2, 0.33, 0.5, 1.0)


def produce(seasons, tag: str, cfg=GBQ, part: tuple = (0, 1)):
    CACHE.mkdir(exist_ok=True)
    rep = Replay(tuple(seasons))
    v2 = make_v2_forecaster(rep, engine.LINEAR)
    pop = pd.read_csv(ROOT / "data/clean/locations.csv", dtype={"location": str}).set_index("location")["population"]
    for i, w in enumerate(rep.weeks):
        if i % part[1] != part[0]:
            continue
        f = CACHE / f"{tag}_{w.ref.date()}.pkl"
        if f.exists():
            continue
        hosp = w.v[~w.v["location"].isin(["US", "11", "72"])]
        a = v2(w)
        b = gbq_forecast(hosp, w.last, pop, cfg)
        f.write_bytes(pickle.dumps({"season": w.season, "ref": w.ref, "last": w.last, "v2": a, "gbq": b}))
        print(w.season, w.ref.date(), "done", flush=True)


def score(seasons, tag: str):
    rep = Replay(tuple(seasons))
    weeks = {w.ref: w for w in rep.weeks}
    store = {}
    for f in sorted(CACHE.glob(f"{tag}_*.pkl")):
        d = pickle.loads(f.read_bytes())
        if d["ref"] in weeks:
            store[d["ref"]] = d

    def forecaster_for(a):
        def fc(w):
            d = store.get(w.ref)
            if d is None:
                return {}
            out = {}
            for k in set(d["v2"]) & set(d["gbq"]):
                q = (1 - a) * d["v2"][k] + a * d["gbq"][k]
                out[k] = np.maximum.accumulate(np.maximum(q, 0))
            return out
        return fc

    rep.weeks = [w for w in rep.weeks if w.ref in store]
    base = rep.score(rep.baseline_forecaster, "baseline")
    res = {a: rep.score(forecaster_for(a), f"blend{a}") for a in BLENDS}
    from backtest.harness import rel
    print(f"{len(rep.weeks)} weeks scored ({tag})")
    for a in BLENDS:
        s = res[a]
        pt, lo, hi = compare(res[0.0], s)
        print(f"GBQ weight {a:<4}: rel {rel(s, base):.4f}  vs v2 {pt:+6.2f}% (95% CI {lo:+.2f} to {hi:+.2f}) "
              f"{verdict(lo, hi):<8} cov50 {s['cov50'].mean():.3f} cov95 {s['cov95'].mean():.3f}")
    return res, base


def variants_with_blend(seasons, tag: str, weight: float, variants: dict):
    """Blend each v2 variant (recomputed; cheap) with the CACHED tree forecasts, and compare it with
    the blend of the base v2 -- so a v2 change is judged as it would actually be used."""
    from backtest.harness import rel
    rep = Replay(tuple(seasons))
    store = {}
    for f in sorted(CACHE.glob(f"{tag}_*.pkl")):
        d = pickle.loads(f.read_bytes())
        store[d["ref"]] = d
    rep.weeks = [w for w in rep.weeks if w.ref in store]
    base = rep.score(rep.baseline_forecaster, "baseline")

    def blend_fc(v2fc):
        def fc(w):
            a, g = v2fc(w), store[w.ref]["gbq"]
            return {k: np.maximum.accumulate(np.maximum((1 - weight) * a[k] + weight * g[k], 0))
                    for k in a if k in g}
        return fc

    ref = rep.score(blend_fc(lambda w: store[w.ref]["v2"]), "blend")
    print(f"{len(rep.weeks)} weeks; base blend (w={weight}) rel {rel(ref, base):.4f} "
          f"cov50 {ref['cov50'].mean():.3f} cov95 {ref['cov95'].mean():.3f}", flush=True)
    for name, cfg in variants.items():
        s = rep.score(blend_fc(make_v2_forecaster(rep, cfg)), name)
        pt, lo, hi = compare(ref, s)
        print(f"{name:<14} rel {rel(s, base):.4f} vs blend {pt:+6.2f}% (95% CI {lo:+.2f} to {hi:+.2f}) "
              f"{verdict(lo, hi):<8} cov50 {s['cov50'].mean():.3f} cov95 {s['cov95'].mean():.3f}", flush=True)


def export_results(seasons, tag: str, weight: float, out: Path, linear_cfg=None):
    """Official results file (same format as run_v2) from cached tree forecasts:
    baseline, v1 (previous design, recomputed), ensemble = linear part blended with the tree model.
    The linear part is recomputed with `linear_cfg` (default engine.LINEAR) -- it's cheap."""
    from backtest.experiments import CURRENT, make_forecaster
    rep = Replay(tuple(seasons))
    store = {}
    for f in sorted(CACHE.glob(f"{tag}_*.pkl")):
        d = pickle.loads(f.read_bytes())
        store[d["ref"]] = d
    missing = [w.ref.date() for w in rep.weeks if w.ref not in store]
    if missing:
        raise SystemExit(f"missing cached weeks for tag {tag}: {missing[:5]}...")

    lin = make_v2_forecaster(rep, linear_cfg or engine.LINEAR)

    def blend(w):
        a, g = lin(w), store[w.ref]["gbq"]
        return {k: np.maximum.accumulate(np.maximum((1 - weight) * a[k] + weight * g[k], 0))
                for k in a if k in g}

    b = rep.score(rep.baseline_forecaster, "baseline")
    v1 = rep.score(make_forecaster(rep, CURRENT), "v1")
    v2 = rep.score(blend, "ensemble")
    key = ["season", "reference_date", "location", "horizon"]
    common = b[key].merge(v1[key], on=key).merge(v2[key], on=key)
    res = pd.concat([d.merge(common, on=key) for d in (b, v1, v2)])
    res["reference_date"] = pd.to_datetime(res["reference_date"]).dt.date
    res.to_csv(out, index=False)
    from backtest.run_v2 import report
    report(res, out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", required=True)
    ap.add_argument("--tag", default="gbq")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--variants", nargs="*", default=None)
    ap.add_argument("--weight", type=float, default=0.33)
    ap.add_argument("--export", type=Path, default=None)
    ap.add_argument("--bags", type=int, default=GBQ.bags, help="tree bags (v2/v3 use 6: --tag gbq6 --bags 6)")
    a = ap.parse_args()
    if a.export:
        export_results(a.seasons, a.tag, a.weight, a.export)
    elif a.variants is not None:
        L = engine.LINEAR
        V = {"loc_scale_05": replace(L, loc_scale=0.5),
             "loc_scale_10": replace(L, loc_scale=1.0),
             "spread_09": replace(L, spread=0.9),
             "resid_bins3": replace(L, resid_bins=3)}
        variants_with_blend(a.seasons, a.tag, a.weight, {k: V[k] for k in (a.variants or V)})
    elif a.score:
        score(a.seasons, a.tag)
    else:
        produce(a.seasons, a.tag, replace(GBQ, bags=a.bags))
