"""Sentinel — step 1: data ingestion.

Downloads the data sources the project needs and saves clean CSVs:
  1. Weekly flu hospital admissions per state (the CDC FluSight hub's own target data).
  2. Weekly share of emergency-department visits that are flu (CDC NSSP, also from the hub).
  3. Every weekly version of the hospital data as first published (the hub's time-series archive).
  4. Daily temperature and humidity per state (Open-Meteo; incremental, history kept in the repo).
  5. CDC NWSS wastewater data for Influenza A (data.cdc.gov dataset ymmh-divb).

Run:  python scripts/ingest.py
Output: data/clean/hosp_flu.csv, data/clean/ed_flu.csv, data/clean/locations.csv,
        data/raw/time-series.csv, data/raw/weather_raw.csv (+ data/history/weather_daily.csv),
        data/raw/wastewater_fluA_raw.csv
"""
from pathlib import Path
import sys
import time

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CLEAN = ROOT / "data" / "clean"
RAW.mkdir(parents=True, exist_ok=True)
CLEAN.mkdir(parents=True, exist_ok=True)

HUB = "https://raw.githubusercontent.com/cdcepi/FluSight-forecast-hub/main"
HOSP_URL = f"{HUB}/target-data/target-hospital-admissions.csv"
ED_URL = f"{HUB}/target-data/target-ed-visits-prop.csv"  # share of ER visits that are flu (CDC NSSP)
LOCATIONS_URL = f"{HUB}/auxiliary-data/locations.csv"
WASTEWATER_API = "https://data.cdc.gov/resource/ymmh-divb.json"  # CDC Wastewater Data for Influenza A


def get_hospital_admissions() -> pd.DataFrame:
    """Weekly lab-confirmed flu hospital admissions, by state (FIPS code) and week ending date."""
    df = pd.read_csv(HOSP_URL, dtype={"location": str})
    df.to_csv(RAW / "target-hospital-admissions.csv", index=False)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["location", "date"]).reset_index(drop=True)
    df = df[["date", "location", "location_name", "value", "weekly_rate"]]
    df.to_csv(CLEAN / "hosp_flu.csv", index=False)
    print(f"Hospital admissions: {len(df):,} rows, {df.location.nunique()} locations, "
          f"{df.date.min().date()} to {df.date.max().date()}")
    return df


def get_ed_visits() -> pd.DataFrame:
    """Weekly share of emergency-department visits that are influenza, by state (a proportion, 0 to 1)."""
    df = pd.read_csv(ED_URL, dtype={"location": str})
    df.to_csv(RAW / "target-ed-visits-prop.csv", index=False)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["location", "date"])[["date", "location", "value"]]
    df.to_csv(CLEAN / "ed_flu.csv", index=False)
    print(f"ER visits: {len(df):,} rows, {df.location.nunique()} locations, "
          f"{df.date.min().date()} to {df.date.max().date()}")
    return df


def get_locations() -> pd.DataFrame:
    loc = pd.read_csv(LOCATIONS_URL, dtype={"location": str})
    loc.to_csv(CLEAN / "locations.csv", index=False)
    return loc


def get_wastewater(page_size: int = 50_000) -> pd.DataFrame | None:
    """Download every row of the CDC Influenza A wastewater dataset (Socrata API, paginated).

    Column names are inspected at runtime and saved as-is in data/raw/; the clean step keeps
    the state, date and concentration-type columns it finds. Check data/raw/wastewater_columns.txt
    the first time and adjust KEEP below if CDC renames columns.
    """
    rows, offset = [], 0
    while True:
        params = {"$limit": page_size, "$offset": offset, "$order": ":id"}
        try:
            r = requests.get(WASTEWATER_API, params=params, timeout=60)
            r.raise_for_status()
        except requests.RequestException as e:
            print(f"Wastewater download failed: {e}", file=sys.stderr)
            return None
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        offset += page_size
        print(f"  wastewater rows so far: {len(rows):,}")
        time.sleep(0.5)  # be polite to the API
    if not rows:
        print("Wastewater download returned no rows; keeping the previous data", file=sys.stderr)
        return None
    ww = pd.DataFrame(rows)
    ww.to_csv(RAW / "wastewater_fluA_raw.csv", index=False)  # same columns as the data.cdc.gov CSV export
    print(f"Wastewater: {len(ww):,} raw sample rows saved (clean them with scripts/wastewater_clean.py)")
    return ww


TIME_SERIES_URL = f"{HUB}/target-data/time-series.csv"  # every weekly version CDC ever published


def get_vintages() -> None:
    """Every weekly version of the hospital (and ER) data as first published. Used to learn how much
    the newest week tends to be revised upward later (reporting delay), see models/engine.py."""
    r = requests.get(TIME_SERIES_URL, timeout=300)
    r.raise_for_status()
    (RAW / "time-series.csv").write_bytes(r.content)
    print(f"Published versions archive: {len(r.content) / 1e6:.0f} MB")


def get_hub_tasks() -> None:
    """The hub's round rules (which Saturdays are rounds, allowed target weeks, submission window),
    used by submission/validate.py --for-submission and scripts/weekly_run.py."""
    r = requests.get(f"{HUB}/hub-config/tasks.json", timeout=60)
    r.raise_for_status()
    (RAW / "hub_tasks.json").write_bytes(r.content)


# one representative point per state (capital or largest metro), same as the backtest weather data
STATE_POINTS = {
    "AL": (32.37, -86.30), "AK": (61.22, -149.90), "AZ": (33.45, -112.07), "AR": (34.75, -92.29),
    "CA": (34.05, -118.24), "CO": (39.74, -104.99), "CT": (41.77, -72.68), "DE": (39.16, -75.52),
    "DC": (38.90, -77.04), "FL": (27.95, -82.46), "GA": (33.75, -84.39), "HI": (21.31, -157.86),
    "ID": (43.62, -116.20), "IL": (41.88, -87.63), "IN": (39.77, -86.16), "IA": (41.60, -93.61),
    "KS": (39.05, -95.69), "KY": (38.25, -85.76), "LA": (29.95, -90.07), "ME": (44.31, -69.78),
    "MD": (39.29, -76.61), "MA": (42.36, -71.06), "MI": (42.33, -83.05), "MN": (44.98, -93.27),
    "MS": (32.30, -90.18), "MO": (38.63, -90.20), "MT": (46.60, -112.04), "NE": (41.26, -95.94),
    "NV": (36.17, -115.14), "NH": (43.21, -71.54), "NJ": (40.22, -74.76), "NM": (35.08, -106.65),
    "NY": (40.71, -74.01), "NC": (35.78, -78.64), "ND": (46.81, -100.78), "OH": (39.96, -82.99),
    "OK": (35.47, -97.52), "OR": (45.52, -122.68), "PA": (40.27, -76.88), "RI": (41.82, -71.41),
    "SC": (34.00, -81.03), "SD": (44.37, -100.35), "TN": (36.16, -86.78), "TX": (29.76, -95.37),
    "UT": (40.76, -111.89), "VT": (44.26, -72.57), "VA": (37.54, -77.44), "WA": (47.61, -122.33),
    "WV": (38.35, -81.63), "WI": (43.07, -89.40), "WY": (41.14, -104.82), "PR": (18.47, -66.11),
}


WEATHER_STORE = ROOT / "data" / "history" / "weather_daily.csv"  # kept in the repo, grows every week


def _fetch_weather(url: str, params: dict, lat: float, lon: float) -> pd.DataFrame | None:
    params = dict(params, latitude=lat, longitude=lon, timezone="UTC",
                  daily="temperature_2m_mean,relative_humidity_2m_mean")
    for attempt in range(6):
        try:
            r = requests.get(url, params=params, timeout=60)
        except requests.RequestException:
            time.sleep(10 * (attempt + 1))
            continue
        if r.status_code == 429:  # Open-Meteo's free-use limit: wait and retry
            time.sleep(15 * (attempt + 1))
            continue
        if r.ok:
            d = r.json()["daily"]
            return pd.DataFrame({"date": d["time"], "temp_c": d["temperature_2m_mean"],
                                 "humidity_pct": d["relative_humidity_2m_mean"]})
        break
    return None


def get_weather(start: str = "2021-10-01") -> None:
    """Daily mean temperature and humidity per state from Open-Meteo (free, no key).

    Incremental: the full daily history is stored in data/history/weather_daily.csv (part of the
    repository), and each run downloads only the last few weeks (archive + the forecast service's
    `past_days`, because the archive runs about 5 days behind). Downloading 5 years for 51 places
    every week would hit Open-Meteo's free-use limits. If the download fails, the stored history is
    used as is: the forecast still runs (the engine carries the last weather week forward up to 2
    weeks)."""
    today = pd.Timestamp.today().normalize()
    store = pd.read_csv(WEATHER_STORE) if WEATHER_STORE.exists() else None
    newest = store.groupby("abbreviation")["date"].max() if store is not None and len(store) else pd.Series(dtype=str)
    arch_end = (today - pd.Timedelta(days=7)).date().isoformat()
    rows, failed = [], []
    for abbr, (lat, lon) in STATE_POINTS.items():
        # each place restarts 3 weeks before ITS OWN newest stored day, so a place that failed for a
        # while catches up instead of keeping a gap
        from_date = ((pd.Timestamp(newest[abbr]) - pd.Timedelta(days=21)).date().isoformat()
                     if abbr in newest.index else start)
        got = False
        # forecast service first, archive second: where both have a day, the archive's value wins
        # (the replay was built on archive values; the forecast service only fills the newest days)
        jobs = [("https://api.open-meteo.com/v1/forecast", {"past_days": 14, "forecast_days": 1}, 0)]
        if from_date <= arch_end:
            jobs.append(("https://archive-api.open-meteo.com/v1/archive",
                         {"start_date": from_date, "end_date": arch_end}, 1))
        for url, params, prio in jobs:
            d = _fetch_weather(url, params, lat, lon)
            if d is not None:
                rows.append(d.assign(abbreviation=abbr, _prio=prio))
                got = True
            time.sleep(1.0)
        if not got:
            failed.append(abbr)
            if len(failed) >= 3 and not rows:  # service down or blocked: stop, use stored history
                failed = list(STATE_POINTS)
                break
    new = (pd.concat(rows).sort_values("_prio").drop(columns="_prio") if rows
           else pd.DataFrame(columns=["abbreviation", "date", "temp_c", "humidity_pct"]))
    w = pd.concat([store, new]) if store is not None else new
    w = w[["abbreviation", "date", "temp_c", "humidity_pct"]].dropna()
    w = w.drop_duplicates(["abbreviation", "date"], keep="last").sort_values(["abbreviation", "date"])
    w = w[pd.to_datetime(w["date"]) < today]
    if w.empty:
        raise SystemExit("No weather data at all (download failed and no stored history).")
    WEATHER_STORE.parent.mkdir(parents=True, exist_ok=True)
    w.to_csv(WEATHER_STORE, index=False)
    w.to_csv(RAW / "weather_raw.csv", index=False)
    msg = f"Weather: {len(w):,} state-days, {w.abbreviation.nunique()} places, up to {w.date.max()}"
    if failed:
        msg += f" (WARNING: download failed for {', '.join(failed)}; using stored history for them)"
    print(msg)


if __name__ == "__main__":
    get_hub_tasks()
    get_locations()
    get_hospital_admissions()
    get_ed_visits()
    get_vintages()
    get_weather()
    get_wastewater()
