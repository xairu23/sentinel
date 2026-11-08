"""Turn the raw CDC Influenza A wastewater file into one weekly signal per state.

Input:  data/raw/wastewater_fluA_raw.csv  (export of data.cdc.gov dataset ymmh-divb)
Output: data/clean/wastewater_state_week.csv
        columns: date (week ending Saturday, same convention as the hub), location (FIPS),
                 ww_level (population-weighted mean of site z-scores), n_sites, pop_covered

Method (simple and explainable):
  1. Use the linear concentration column (pcr_target_avg_conc_lin, non-detects already set to LOD/2 by CDC).
  2. Each site measures in its own units and lab method, so compare a site only with itself:
     log10(concentration), z-scored against that site's own PAST samples (no look-ahead).
  3. Average each site's samples within a week (Sunday-Saturday, labelled by the Saturday).
  4. State value = population-weighted mean of its sites that week.
"""
from pathlib import Path
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "wastewater_fluA_raw.csv"
OUT = ROOT / "data" / "clean" / "wastewater_state_week.csv"
LOC = ROOT / "data" / "clean" / "locations.csv"


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.replace(",", ""), errors="coerce")


def main() -> pd.DataFrame | None:
    if not RAW.exists():
        # the download failed (scripts/ingest.py says why): keep the last cleaned file, which is part
        # of the repository. The engine uses wastewater one week behind and bridges at most 2 missing
        # weeks; beyond that the regressions without wastewater carry the forecast.
        print("WARNING: no raw wastewater file; keeping the existing data/clean/wastewater_state_week.csv")
        return None
    age_days = (time.time() - RAW.stat().st_mtime) / 86400
    if age_days > 6:
        print(f"WARNING: the raw wastewater file is {age_days:.0f} days old (today's download failed?); "
              "cleaning it anyway")
    ww = pd.read_csv(RAW, low_memory=False, dtype=str)
    ww["conc"] = num(ww["pcr_target_avg_conc_lin"])
    ww["pop"] = num(ww["population_served"])
    ww["date"] = pd.to_datetime(ww["sample_collect_date"], errors="coerce")
    ww = ww.dropna(subset=["conc", "date", "state_territory"])
    ww = ww[ww["conc"] > 0]

    ww["log_conc"] = np.log10(ww["conc"])
    # z-score each sample against that site's PAST samples only (expanding window, >= 20 samples),
    # so no future information leaks into the signal used for backtesting.
    ww = ww.sort_values(["site", "pcr_target_units", "date"])
    g = ww.groupby(["site", "pcr_target_units"])["log_conc"]
    past_mean = g.transform(lambda s: s.shift(1).expanding(min_periods=20).mean())
    past_std = g.transform(lambda s: s.shift(1).expanding(min_periods=20).std())
    ww["z"] = (ww["log_conc"] - past_mean) / past_std.where(past_std > 0)
    ww = ww.replace([np.inf, -np.inf], np.nan).dropna(subset=["z"])
    ww["z"] = ww["z"].clip(-4, 4)  # cap extreme values from sites with short or unstable histories

    # week ending Saturday (hub reference dates are Saturdays)
    ww["week"] = ww["date"] + pd.to_timedelta((5 - ww["date"].dt.dayofweek) % 7, unit="D")

    site_week = (ww.groupby(["state_territory", "site", "week"])
                   .agg(z=("z", "mean"), pop=("pop", "max")).reset_index())
    site_week["pop"] = site_week["pop"].fillna(site_week["pop"].median())

    def wmean(d):
        return pd.Series({"ww_level": np.average(d["z"], weights=d["pop"]),
                          "n_sites": d["site"].nunique(), "pop_covered": d["pop"].sum()})

    state = site_week.groupby(["state_territory", "week"]).apply(wmean, include_groups=False).reset_index()

    # map two-letter state code -> FIPS using the hub's locations file
    loc = pd.read_csv(LOC, dtype={"location": str})
    abbr_col = next(c for c in loc.columns if c.lower() in ("abbreviation", "abbr", "state_abbr"))
    m = dict(zip(loc[abbr_col].str.lower(), loc["location"]))
    state["location"] = state["state_territory"].str.lower().map(m)
    state = state.dropna(subset=["location"]).rename(columns={"week": "date"})
    state = state[["date", "location", "state_territory", "ww_level", "n_sites", "pop_covered"]]
    state = state.sort_values(["location", "date"])
    state.to_csv(OUT, index=False)
    print(f"Wastewater state-weeks: {len(state):,} | states: {state.location.nunique()} | "
          f"{state.date.min().date()} to {state.date.max().date()} | raw samples used: {len(ww):,}")
    return state


if __name__ == "__main__":
    main()
