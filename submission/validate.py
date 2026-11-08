"""Check a submission file against the CDC FluSight hub rules BEFORE it is sent.

Run:  python -m submission.validate submission/output/2026-10-17-Sentinel-Ens.csv [--for-submission]
Exits with an error (and sends nothing) if any rule fails. The hub also runs its own official
checks (hubValidations) on every pull request; this is our own safety net so a broken file never leaves.
"""
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from backtest.wis import QUANTILES
from submission import hub_rules

COLUMNS = ["reference_date", "target", "horizon", "target_end_date", "location",
           "output_type", "output_type_id", "value"]
LOCATIONS_FILE = Path(__file__).resolve().parents[1] / "data/clean/locations.csv"


def validate(path: Path, for_submission: bool = False) -> list[str]:
    """for_submission: also check the hub's round rules (a real round, allowed target weeks, and
    today, US Eastern time, inside the round's submission window)."""
    errors = []
    name = path.name
    d = pd.read_csv(path, dtype={"location": str, "output_type_id": str})
    if list(d.columns) != COLUMNS:
        errors.append(f"columns must be exactly {COLUMNS}, got {list(d.columns)}")
        return errors
    ref = d["reference_date"].unique()
    if len(ref) != 1:
        errors.append("file must contain exactly one reference_date")
    elif not name.startswith(ref[0]):
        errors.append(f"file name must start with the reference date {ref[0]}")
    elif pd.Timestamp(ref[0]).dayofweek != 5:
        errors.append("reference_date must be a Saturday")
    if not set(d["target"]) <= {"wk inc flu hosp", "wk flu hosp rate change"}:
        errors.append("only targets 'wk inc flu hosp' and 'wk flu hosp rate change' are produced")
    if "wk inc flu hosp" not in set(d["target"]):
        errors.append("the required target 'wk inc flu hosp' is missing")
    exp_end = pd.to_datetime(d["reference_date"]) + pd.to_timedelta(7 * d["horizon"], unit="D")
    if (pd.to_datetime(d["target_end_date"]) != exp_end).any():
        errors.append("target_end_date must equal reference_date + 7 * horizon")
    valid_locs = set(pd.read_csv(LOCATIONS_FILE, dtype={"location": str})["location"])
    bad = set(d["location"]) - valid_locs
    if bad:
        errors.append(f"unknown location codes: {sorted(bad)[:5]}")
    if d["value"].isna().any() or (d["value"] < 0).any():
        errors.append("values must be present and non-negative")
    if d.duplicated(["target", "horizon", "location", "output_type", "output_type_id"]).any():
        errors.append("duplicate rows")

    # 1. weekly admissions: 23 quantiles, whole numbers, non-decreasing, plausible
    q = d[d["target"] == "wk inc flu hosp"]
    if set(q["output_type"]) != {"quantile"}:
        errors.append("'wk inc flu hosp' rows must have output_type 'quantile'")
    if not set(q["horizon"]).issubset({-1, 0, 1, 2, 3}):
        errors.append("'wk inc flu hosp' horizons must be in -1..3")
    if (q["value"] != q["value"].round()).any():
        # hub rule since the 2025-26 season: weekly admissions forecasts are whole numbers
        errors.append("'wk inc flu hosp' values must be integers")
    pop = pd.read_csv(LOCATIONS_FILE, dtype={"location": str}).set_index("location")["population"]
    over = q[q["value"] > 0.3 * q["location"].map(pop)]
    if len(over):
        # hub plausibility bound (hub-config/validations.yml, max_hosp_popn_frac): <= 30% of population
        errors.append(f"values above 30% of the population (hub plausibility bound): {sorted(set(over['location']))[:5]}")
    expected_q = {f"{x:g}" for x in QUANTILES}
    for (loc, h), g in q.groupby(["location", "horizon"]):
        if set(g["output_type_id"]) != expected_q or len(g) != len(QUANTILES):
            errors.append(f"{loc} h{h}: must have exactly the 23 required quantiles")
            continue
        g = g.assign(qq=g["output_type_id"].astype(float)).sort_values("qq")
        if (g["value"].diff().dropna() < 0).any():
            errors.append(f"{loc} h{h}: values must not decrease as the quantile increases")

    if for_submission:
        tasks = hub_rules.load_tasks()
        if tasks is None:
            errors.append("hub rules (tasks.json) not found: run scripts/ingest.py first")
        elif len(ref) == 1:
            r0 = date.fromisoformat(ref[0])
            if ref[0] not in hub_rules.reference_dates(tasks):
                errors.append(f"{ref[0]} is not a forecast round of the hub")
            bad_end = set(d["target_end_date"]) - hub_rules.target_end_dates(tasks)
            if bad_end:
                errors.append(f"target weeks not accepted by the hub: {sorted(bad_end)[:3]}")
            start, end = hub_rules.window(tasks, r0)
            today = hub_rules.now_eastern().date()
            if not start <= today <= end:
                errors.append(f"today ({today}, US Eastern) is outside the submission window {start} to {end}")

    # 2. optional trend target: 5 category probabilities summing to 1, horizons 0..3
    r = d[d["target"] == "wk flu hosp rate change"]
    if len(r):
        cats = {"large_decrease", "decrease", "stable", "increase", "large_increase"}
        if set(r["output_type"]) != {"pmf"}:
            errors.append("'wk flu hosp rate change' rows must have output_type 'pmf'")
        if not set(r["horizon"]).issubset({0, 1, 2, 3}):
            errors.append("'wk flu hosp rate change' horizons must be in 0..3")
        if (r["value"] > 1).any():
            errors.append("trend probabilities must be between 0 and 1")
        for (loc, h), g in r.groupby(["location", "horizon"]):
            if set(g["output_type_id"]) != cats or len(g) != 5:
                errors.append(f"{loc} h{h}: trend must have exactly the 5 categories")
            elif abs(g["value"].sum() - 1) > 1e-6:
                errors.append(f"{loc} h{h}: trend probabilities must sum to 1 (got {g['value'].sum():.6f})")
    return errors


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    f = Path(args[0])
    errs = validate(f, for_submission="--for-submission" in sys.argv)
    if errs:
        print("INVALID:\n- " + "\n- ".join(errs[:20]))
        sys.exit(1)
    print(f"VALID: {f.name}" + (" (hub round rules and submission window checked)" if "--for-submission" in sys.argv else ""))
