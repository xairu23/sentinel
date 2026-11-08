"""Sentinel — lab-test positivity signal. Turns CDC/Delphi clinical-lab flu test results
(% of flu tests that came back positive, by state and week) into one weekly number per state,
on the same log scale used by the other signals.

Source: CMU Delphi `fluview_clinical` endpoint (CDC's own clinical-lab surveillance, same family
of data as FluView). Columns: region (postal abbreviation), epiweek (YYYYWW), total_specimens,
percent_positive.

Run: python scripts/lab_clean.py
Input:  data/raw/lab_positivity_raw.csv
Output: data/clean/lab_state_week.csv  (date, location, lab_level)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MIN_SPECIMENS = 20  # weeks with very few tests give a noisy percentage; drop them


def epiweek_to_saturday(epiweek: pd.Series) -> pd.Series:
    """CDC epiweek YYYYWW -> the Saturday ending that week (same convention as the hub data)."""
    year = (epiweek // 100).astype(int)
    week = (epiweek % 100).astype(int)
    jan4 = pd.to_datetime(year.astype(str) + "-01-04")  # ISO week 1 always contains Jan 4
    week1_sat = jan4 - pd.to_timedelta(jan4.dt.weekday + 1, unit="D") + pd.Timedelta(days=6)
    return week1_sat + pd.to_timedelta((week - 1) * 7, unit="D")


def main() -> None:
    raw = pd.read_csv(ROOT / "data/raw/lab_positivity_raw.csv")
    raw = raw[raw["region"].str.len() == 2].copy()  # drop "region"/multi-state rollup rows
    raw["abbreviation"] = raw["region"].str.upper()
    raw = raw[raw["total_specimens"] >= MIN_SPECIMENS]

    loc = pd.read_csv(ROOT / "data/clean/locations.csv", dtype={"location": str})
    raw = raw.merge(loc[["abbreviation", "location"]], on="abbreviation", how="inner")

    raw["date"] = epiweek_to_saturday(raw["epiweek"])
    # log scale like the other signals; floor so a true 0% doesn't become -inf
    raw["lab_level"] = np.log(raw["percent_positive"].clip(lower=0.05) / 100)

    out = raw.groupby(["date", "location"], as_index=False)["lab_level"].mean()
    out = out.sort_values(["location", "date"])
    out.to_csv(ROOT / "data/clean/lab_state_week.csv", index=False)
    print(f"Wrote data/clean/lab_state_week.csv: {len(out):,} rows, "
          f"{out['location'].nunique()} states, {out['date'].min().date()} to {out['date'].max().date()}")


if __name__ == "__main__":
    main()
