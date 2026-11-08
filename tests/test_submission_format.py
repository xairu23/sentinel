"""Checks for the submission file format and the trend target.   Run: python -m tests.test_submission_format

1. The trend categories computed by submission/rate_change.py match the CDC hub's own "oracle"
   categories for every past week (target-data/oracle-output.csv), so the rules are implemented right.
2. Trend probabilities are valid for awkward inputs (zero counts, tiny places, tied quantiles).
3. A file with both targets, built the way submission/build_submission.py builds it, passes
   submission/validate.py; and the validator catches typical mistakes.
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.wis import QUANTILES
from submission.rate_change import CATEGORIES, category, pmf
from submission.validate import validate

ROOT = Path(__file__).resolve().parents[1]
HUB = ROOT.parent / "FluSight-forecast-hub"


def check_oracle() -> int:
    f = HUB / "target-data/oracle-output.csv"
    if not f.exists():
        print("oracle check skipped (no hub copy)")
        return 0
    o = pd.read_csv(f, dtype={"location": str, "output_type_id": str}, low_memory=False)
    o = o[(o["target"] == "wk flu hosp rate change") & (o["oracle_value"] == 1)]
    o["target_end_date"] = pd.to_datetime(o["target_end_date"])
    t = pd.read_csv(HUB / "target-data/target-hospital-admissions.csv", dtype={"location": str},
                    parse_dates=["date"]).set_index(["location", "date"])["value"]
    pop = pd.read_csv(ROOT / "data/clean/locations.csv", dtype={"location": str}).set_index("location")["population"]
    bad = n = 0
    for r in o.itertuples():
        b = t.get((r.location, r.target_end_date - pd.Timedelta(weeks=r.horizon + 1)))
        y = t.get((r.location, r.target_end_date))
        if b is None or y is None or np.isnan(b) or np.isnan(y) or r.location not in pop:
            continue
        n += 1
        bad += category(b, y, pop[r.location], r.horizon) != r.output_type_id
    print(f"trend categories vs the hub's oracle: {n - bad:,} of {n:,} identical")
    return bad


def check_pmf_edges() -> int:
    bad = 0
    cases = [(np.zeros(23), 0.0, 580_000, 0), (np.arange(23.0), 3.0, 50_000, 3),
             (np.full(23, 120.0), 120.0, 39_000_000, 1), (np.linspace(500, 5000, 23), 2500.0, 330_000_000, 2)]
    for q, base, pop, h in cases:
        p = pmf(q, base, pop, h)
        v = np.array([p[c] for c in CATEGORIES])
        if set(p) != set(CATEGORIES) or (v < 0).any() or (v > 1).any() or abs(v.sum() - 1) > 1e-9:
            bad += 1
            print("bad pmf:", q[:3], base, pop, h, p)
    print(f"trend probabilities for awkward inputs: {len(cases) - bad} of {len(cases)} valid")
    return bad


def check_file() -> int:
    src = ROOT / "submission/sealed/2026-10-03-Sentinel-Ens.csv"
    d = pd.read_csv(src, dtype={"location": str, "output_type_id": str})
    pop = pd.read_csv(ROOT / "data/clean/locations.csv", dtype={"location": str}).set_index("location")["population"]
    rows = []
    for (loc, h, end), g in d.groupby(["location", "horizon", "target_end_date"]):
        q = g.assign(t=g["output_type_id"].astype(float)).sort_values("t")["value"].values.astype(float)
        for cat, v in pmf(q, float(q[11]), float(pop[loc]), int(h)).items():
            rows.append({"reference_date": g["reference_date"].iloc[0], "target": "wk flu hosp rate change",
                         "horizon": h, "target_end_date": end, "location": loc, "output_type": "pmf",
                         "output_type_id": cat, "value": float(v)})
    full = pd.concat([d, pd.DataFrame(rows)])
    bad = 0
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / src.name
        full.to_csv(f, index=False)
        errs = validate(f)
        print(f"file with both targets ({len(full):,} rows): {'VALID' if not errs else errs[:3]}")
        bad += bool(errs)
        # the validator must catch these
        for label, broken in [
            ("decimal admissions", full.assign(value=np.where(full["target"] == "wk inc flu hosp", full["value"] + 0.5, full["value"]))),
            ("trend not summing to 1", full.assign(value=np.where(full["output_type_id"] == "stable", full["value"] + 0.1, full["value"]))),
            ("missing quantile", full[full["output_type_id"] != "0.5"]),
        ]:
            broken.to_csv(f, index=False)
            caught = bool(validate(f))
            print(f"validator catches {label}: {caught}")
            bad += not caught
    return bad


if __name__ == "__main__":
    total = check_oracle() + check_pmf_edges() + check_file()
    print("PASS" if total == 0 else f"FAIL ({total})")
