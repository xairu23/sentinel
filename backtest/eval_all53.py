"""Scores of the all-53-places replay (the exact live setup: training on all 53 places), including
the US, DC and Puerto Rico, which the 50-state experiments leave out. Compared with the CDC's own
baseline and ensemble on identical forecasts, on counts and on log counts.

    python -m backtest.exp_v3 --produce-all      # the caches (hours)
    python -m backtest.eval_all53                # this report -> backtest/all53.md
"""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.exp_v3 import CACHE, SEASONS, blend, quantile_average
from backtest.leaderboard import load_model
from backtest.member_cache import mean_members
from backtest.run_backtest import final_hosp
from backtest.wis import QUANTILES
from models.engine import V3

ROOT = Path(__file__).resolve().parents[1]
HUB = ROOT.parent / "FluSight-forecast-hub"
SPECIAL = {"US": "United States", "11": "District of Columbia", "72": "Puerto Rico"}


def load_family(fam: str, tag: str = "all53") -> dict:
    out = {}
    for f in sorted(CACHE.glob(f"{tag}{fam}_*.pkl")):
        d = pickle.loads(f.read_bytes())
        out[d["ref"]] = d
    return out


def wis_rows(Q: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 2.0 * ((((y[:, None] <= Q).astype(float)) - QUANTILES[None, :]) * (Q - y[:, None])).mean(axis=1)


def forecasts() -> tuple[dict, dict]:
    mem, gbq, qar = load_family("mem"), load_family("gbq"), load_family("qar")
    refs = sorted(set(mem) & set(gbq) & set(qar))
    season = {r: mem[r]["season"] for r in refs}
    lin = {r: mean_members(mem[r]["mem"]) for r in refs}
    g = {r: gbq[r]["gbq"] for r in refs}
    q = {r: qar[r]["qar"] for r in refs}
    models = {
        "regressions": lin, "trees": g, "QAR": q,
        "v2": {r: blend(lin[r], g[r], 0.33) for r in refs},
        # v3 as submitted: Puerto Rico without QAR (idea 9, engine.V3.qar_skip)
        "v3": {r: quantile_average([lin[r], g[r], {k: v for k, v in q[r].items() if k[0] not in V3.qar_skip}])
               for r in refs},
        "v3 before idea 9": {r: quantile_average([lin[r], g[r], q[r]]) for r in refs},
    }
    return models, season


def main():
    models, season = forecasts()
    refs = sorted(season)
    truth = final_hosp().set_index(["location", "date"])["value"]
    rows = []
    for r in refs:
        last = r - pd.Timedelta(weeks=1)
        for (loc, k) in models["v3"][r]:
            y = truth.get((loc, last + pd.Timedelta(weeks=k)))
            if y is not None and not np.isnan(y):
                rows.append((season[r], r, str(r.date()), loc, k - 1, float(y)))
    keys = pd.DataFrame(rows, columns=["season", "ref", "ref_s", "location", "horizon", "y"])
    y = keys["y"].values
    Q = {m: np.round(np.vstack([p[r][(l, h + 1)] for r, l, h in zip(keys.ref, keys.location, keys.horizon)]))
         for m, p in models.items()}
    idx = pd.MultiIndex.from_frame(keys[["ref_s", "location", "horizon"]])
    kdf = keys.rename(columns={"ref_s": "ref"})[["ref", "location", "horizon"]]
    for m in ("FluSight-baseline", "FluSight-ensemble"):
        w = load_model(HUB / "model-output" / m, kdf)
        w.index = w.index.set_names(["ref_s", "location", "horizon"])
        Q[m] = w.reindex(idx).values.astype(float)
    ok = ~np.isnan(Q["FluSight-baseline"]).any(axis=1) & ~np.isnan(Q["FluSight-ensemble"]).any(axis=1)
    keys, y = keys[ok].reset_index(drop=True), y[ok]
    Q = {m: v[ok] for m, v in Q.items()}
    W = {sc: {m: wis_rows(f(v), f(y)) for m, v in Q.items()}
         for sc, f in (("counts", lambda a: a), ("log", lambda a: np.log1p(np.maximum(a, 0))))}

    def rel(sc, m, mask):
        return W[sc][m][mask].sum() / W[sc]["FluSight-baseline"][mask].sum()

    order = ["v3", "v3 before idea 9", "v2", "regressions", "trees", "QAR", "FluSight-ensemble"]
    lines = ["# All 53 places: the live setup, scored", "",
             f"{len(keys):,} forecasts ({keys['ref'].nunique()} weeks), every place the hub scores, compared with the CDC's",
             "own baseline and ensemble on identical forecasts. Relative WIS = total WIS / CDC baseline's total WIS",
             "(below 1 = better). Sentinel quantiles rounded to whole numbers as submitted. v3 is in-sample on all",
             "seasons; v2 is blind on 2025-26.", ""]
    for sc in ("log", "counts"):
        t = pd.DataFrame({m: [rel(sc, m, (keys["season"] == s).values) for s in SEASONS] +
                          [rel(sc, m, np.ones(len(keys), bool))] for m in order},
                         index=list(SEASONS) + ["all"]).T.round(3)
        lines += [f"## All 53 places, {sc} scale", "", t.to_markdown(), ""]
    for loc, name in SPECIAL.items():
        m_loc = (keys["location"] == loc).values
        t = pd.DataFrame({m: [rel(sc, m, m_loc & (keys["season"] == s).values) for s in SEASONS] +
                          [rel(sc, m, m_loc)] for m in order for sc in ("log",)},
                         index=list(SEASONS) + ["all"]).T.round(3)
        lines += [f"## {name} ({loc}), log scale", "", t.to_markdown(), ""]
    st = ~keys["location"].isin(list(SPECIAL)).values
    lines += ["## The 50 states in the live setup vs the 50-state replay", "",
              f"v3 relative WIS on counts, live setup (trained on 53 places): {rel('counts', 'v3', st):.3f}; "
              f"see results_v3.csv for the 50-state replay (trained on 50).", ""]
    for m in ("v3", "v2"):
        Qm = Q[m]
        cov = ((Qm[:, 1] <= y) & (y <= Qm[:, 21]))
        lines.append(f"- {m}: 95% range held the truth {cov.mean():.0%} overall; "
                     + ", ".join(f"{SPECIAL[l]} {cov[(keys['location'] == l).values].mean():.0%}" for l in SPECIAL))
    (ROOT / "backtest/all53.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
