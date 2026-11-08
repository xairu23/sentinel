"""Is Sentinel's 'rate change' (trend category) forecast any good? Replay check before submitting it.

For every replay week and state, the 5 category probabilities are derived from the v3 quantiles
(submission/rate_change.py) and from the flat baseline's quantiles in exactly the same way, then
scored against the category that really happened (final data for both weeks, as the CDC scores it).
The 2026-27 thresholds are used for every season.

    python -m backtest.rate_change_eval
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backtest.exp_v3 import Book, quantile_average
from backtest.harness import Replay
from backtest.member_cache import mean_members
from submission.rate_change import CATEGORIES, category, pmf

ROOT = Path(__file__).resolve().parents[1]


def main():
    book = Book()
    rep = Replay(("2023-24", "2024-25", "2025-26"))
    pop = pd.read_csv(ROOT / "data/clean/locations.csv", dtype={"location": str}).set_index("location")["population"]
    final = book.truth
    rows = []
    for w in rep.weeks:
        lin = mean_members(book.mem[w.ref]["members"])
        v3 = quantile_average([lin, book.gbq[w.ref], book.qar[w.ref]])
        base_q = rep.baseline_forecaster(w)
        pub = w.v.set_index(["location", "date"])["value"]
        for (loc, k), q in v3.items():
            if loc in ("US", "11", "72") or (loc, k) not in base_q:
                continue
            h = k - 1
            b_pub = pub.get((loc, w.last))
            b_fin, y = final.get((loc, w.last)), final.get((loc, w.last + pd.Timedelta(weeks=k)))
            if any(v is None or np.isnan(v) for v in (b_pub, b_fin, y)):
                continue
            truth = category(b_fin, y, pop[loc], h)
            for model, qq in (("v3", np.round(q)), ("baseline", np.round(base_q[(loc, k)]))):
                p = pmf(qq, b_pub, pop[loc], h)
                vec = np.array([p[c] for c in CATEGORIES])
                onehot = np.array([c == truth for c in CATEGORIES], float)
                rows.append({"season": w.season, "ref": w.ref, "location": loc, "horizon": h, "model": model,
                             "truth": truth, "brier": float(((vec - onehot) ** 2).sum()),
                             "logscore": float(-np.log(max(p[truth], 1e-3))),
                             "p_true": p[truth], "top": CATEGORIES[int(np.argmax(vec))] == truth,
                             **{f"p_{c}": p[c] for c in CATEGORIES}})
    r = pd.DataFrame(rows)
    r.to_csv(ROOT / "backtest/rate_change_eval.csv", index=False)
    print("Mean Brier score (lower = better), by season:")
    print(r.pivot_table(index="season", columns="model", values="brier").round(3).to_string())
    print("\nMean log score (lower = better), by horizon:")
    print(r.pivot_table(index="horizon", columns="model", values="logscore").round(3).to_string())
    print("\nMost likely category was right:", r.groupby("model")["top"].mean().round(3).to_dict())
    v = r[r.model == "v3"]
    cal = pd.DataFrame({c: [v[f"p_{c}"].mean(), (v["truth"] == c).mean()] for c in CATEGORIES},
                       index=["v3 mean probability", "observed share"]).round(3)
    print("\nCalibration (v3):")
    print(cal.to_string())
    compare_hub(r)


def compare_hub(r: pd.DataFrame, models=("FluSight-ensemble", "UMass-flusion")):
    """Brier score of the hub teams' own trend forecasts vs Sentinel's, on identical forecasts.
    2023-24 is left out: its category thresholds were different (hub README)."""
    hub = ROOT.parent / "FluSight-forecast-hub/model-output"
    v = r[r["model"] == "v3"].set_index(["ref", "location", "horizon"])
    out = []
    for m in models:
        parts = []
        for f in sorted((hub / m).glob("*.csv")):
            d = pd.read_csv(f, dtype={"location": str, "output_type_id": str})
            d = d[d["target"] == "wk flu hosp rate change"]
            if len(d):
                parts.append(d)
        if not parts:
            continue
        d = pd.concat(parts)
        d["reference_date"] = pd.to_datetime(d["reference_date"])
        w = d.pivot_table(index=["reference_date", "location", "horizon"], columns="output_type_id", values="value")
        common = v.index.intersection(w.index)
        vv, P = v.loc[common], w.loc[common, CATEGORIES].values
        P = P / P.sum(axis=1, keepdims=True)
        onehot = np.vstack([[c == t for c in CATEGORIES] for t in vv["truth"]]).astype(float)
        bm = ((P - onehot) ** 2).sum(axis=1)
        for s in ("2024-25", "2025-26"):
            mk = (vv["season"] == s).values
            if mk.sum():
                out.append({"team": m, "season": s, "forecasts": int(mk.sum()),
                            "team Brier": round(bm[mk].mean(), 3), "Sentinel v3 Brier": round(vv["brier"].values[mk].mean(), 3)})
    print("\nAgainst the hub teams' own trend forecasts (same forecasts, lower = better):")
    print(pd.DataFrame(out).to_string(index=False))


if __name__ == "__main__":
    main()
