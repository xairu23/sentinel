"""The last three months (July - September 2026): what v1, v2 and v3 would have forecast each week,
versus what really happened, in numbers of patients.

Each week uses only data up to that week (everything later is cut off). Honest limits:
  * CDC's archive has the as-first-published hospital numbers for only 2 of these 13 weeks, so this
    check uses today's corrected numbers for all versions alike (the reporting-delay fix is switched
    off, since there is nothing to correct). That is a bit optimistic for every version.
  * Experiment 5 (the 30% bias correction, in v1, v2 and v3) was motivated by looking at these same
    months, so none of the versions is fully blind here.

    python -m backtest.recent_check
Writes backtest/recent_check.csv and backtest/recent_check.md
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.experiments import CURRENT, cols_for, fit_one, members as v1_members
from backtest.run_backtest import attach_signals, final_hosp, load_signals
from backtest.wis import QUANTILES
from models import engine
from models.gbq import GBQ, gbq_forecast
from models.models import baseline, load_neighbor_pairs, make_features
from models.qar import qar_forecast

ROOT = Path(__file__).resolve().parents[1]
REFS = pd.date_range("2026-07-04", "2026-09-26", freq="7D")


def wis(q: np.ndarray, y: float) -> float:
    return float(2 * (((y <= q).astype(float) - QUANTILES) * (q - y)).mean())


def main():
    load_neighbor_pairs(ROOT / "data/clean/locations.csv")
    sig = load_signals()
    fh = final_hosp()
    truth = fh.set_index(["location", "date"])["value"]
    pop = pd.read_csv(ROOT / "data/clean/locations.csv", dtype={"location": str}).set_index("location")["population"]
    cfg = replace(engine.LINEAR, backfill=0)
    rows = []
    for ref in REFS:
        last = ref - pd.Timedelta(weeks=1)
        h = fh[fh["date"] <= last]
        s = {k: sig[k][sig[k]["date"] <= last] for k in ("ww", "ed", "weather")}
        # v1: the first design (signals one week behind)
        p1 = attach_signals(h, sig, keep_ww_states=False)
        f1 = make_features(p1)
        v1m = [fit_one(f1, last, cols_for(c, CURRENT), CURRENT) for c in v1_members(CURRENT)]
        # v2 and v3 share the regressions and the trees (same code paths as models/engine.py)
        mem = engine.member_forecasts(h, last, last, s, None, pop, cfg)
        lin = {}
        for key in set().union(*mem.values()):
            qs = [p[key] for p in mem.values() if key in p]
            lin[key] = np.maximum.accumulate(np.maximum(np.mean(qs, axis=0), 0))
        g = gbq_forecast(h, last, pop, replace(GBQ, bags=6))
        q = qar_forecast(h, last, pop)
        for loc in sorted(fh["location"].unique()):
            series = h[h["location"] == loc].set_index("date")["value"].dropna()
            for k in (1, 2, 3, 4):
                tdate = last + pd.Timedelta(weeks=k)
                y = truth.get((loc, tdate))
                if y is None or np.isnan(y) or (loc, k) not in lin:
                    continue
                qs1 = [p[(loc, k)] for p in v1m if (loc, k) in p]
                if not qs1 or len(series) < 30:
                    continue
                out = {"v1": np.maximum.accumulate(np.maximum(np.mean(qs1, axis=0), 0)),
                       "baseline": baseline(series, k)}
                a = lin[(loc, k)]
                out["v2"] = np.maximum.accumulate(np.maximum((1 - 0.33) * a + 0.33 * g[(loc, k)], 0)) if (loc, k) in g else a
                comps = [a] + [c[(loc, k)] for c in (g, q) if (loc, k) in c
                               and not (c is q and loc in engine.V3.qar_skip)]
                out["v3"] = np.maximum.accumulate(np.maximum(np.mean(comps, axis=0), 0))
                for m, qq in out.items():
                    qq = np.round(qq)
                    rows.append({"made_with_data_to": last.date(), "target_week": tdate.date(), "location": loc,
                                 "weeks_ahead": k, "model": m, "actual": y, "median": qq[11],
                                 "lo95": qq[1], "hi95": qq[21], "wis": wis(qq, y)})
        print(ref.date(), "done", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(ROOT / "backtest/recent_check.csv", index=False)
    report(r)


def report(r: pd.DataFrame):
    lines = ["# July - September 2026: forecasts vs what happened", "",
             "Each forecast uses only data up to the week before its target (1 week ahead) or earlier.",
             "Today's corrected hospital numbers are used for every version (CDC archived the first-published",
             "numbers for only 2 of these weeks), so all versions look a bit better than they would have live.",
             "The 30% bias correction in all versions was designed while looking at these months.", ""]
    us = r[(r["location"] == "US") & (r["weeks_ahead"] == 1)]
    t = us.pivot_table(index="target_week", columns="model", values="median").join(
        us.groupby("target_week")["actual"].first())
    v3 = us[us["model"] == "v3"].set_index("target_week")
    t["v3 95% range"] = v3["lo95"].astype(int).astype(str) + " - " + v3["hi95"].astype(int).astype(str)
    t = t[["actual", "v1", "v2", "v3", "v3 95% range"]]
    lines += ["## Whole US, 1 week ahead (patients admitted with flu that week)", "",
              t.astype({c: int for c in ("actual", "v1", "v2", "v3")}).to_markdown(), ""]
    stats = []
    for m in ("v1", "v2", "v3"):
        d = us[us["model"] == m]
        e = (d["median"] - d["actual"]) / d["actual"] * 100
        cov = ((d["lo95"] <= d["actual"]) & (d["actual"] <= d["hi95"])).mean()
        stats.append({"model": m, "mean absolute error %": round(e.abs().mean(), 1),
                      "mean signed error %": round(e.mean(), 1), "95% range held the truth": f"{cov:.0%}"})
    lines += [pd.DataFrame(stats).to_markdown(index=False), ""]
    st = r[~r["location"].isin(["US", "11", "72"])]
    w = st.pivot_table(index="weeks_ahead", columns="model", values="wis", aggfunc="sum")
    rel = w[["v1", "v2", "v3"]].div(w["baseline"], axis=0).round(3)
    tot = (w[["v1", "v2", "v3"]].sum() / w["baseline"].sum()).round(3)
    rel.loc["all"] = tot
    lines += ["## 50 states: WIS relative to Sentinel's baseline (lower = better)", "",
              rel.to_markdown(), ""]
    (ROOT / "backtest/recent_check.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
