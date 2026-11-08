"""Sentinel — weather signal. Turns daily state-level temperature and humidity (Open-Meteo, one
representative point per state) into one weekly cold/dry-ness number per state.

Cold, dry air is strongly associated with flu transmission. The model already has generic
sin/cos time-of-year terms; this lets each state's actual weather (not just the calendar date)
tell the model how "flu-favourable" conditions are that week, which can differ by region and
by year (an unusually warm or dry week in a state that's normally cold by now).

Run: python scripts/weather_clean.py
Input:  data/raw/weather_raw.csv       (abbreviation, date, temp_c, humidity_pct — daily)
Output: data/clean/weather_state_week.csv  (date, location, weather_level)
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main(signal: str = "index") -> None:
    w = pd.read_csv(ROOT / "data/raw/weather_raw.csv", parse_dates=["date"])
    w["abbreviation"] = w["abbreviation"].str.upper()

    loc = pd.read_csv(ROOT / "data/clean/locations.csv", dtype={"location": str})
    w = w.merge(loc[["abbreviation", "location"]], on="abbreviation", how="inner")

    # week ending Saturday, same convention as the other signals
    w["week"] = w["date"] + pd.to_timedelta((5 - w["date"].dt.weekday) % 7, unit="D")

    t, rh = w["temp_c"], w["humidity_pct"]
    w["ah"] = 6.112 * np.exp(17.67 * t / (t + 243.5)) * rh * 2.1674 / (273.15 + t)
    wk = w.groupby(["location", "week"], as_index=False).agg(temp_c=("temp_c", "mean"),
                                                               humidity_pct=("humidity_pct", "mean"),
                                                               ah=("ah", "mean"))
    # one "flu-favourable" number: cold AND dry both push it up. Standardise each state's own
    # temperature and humidity against ITS OWN PAST ONLY (expanding window, >= 52 weeks so a full
    # year of seasonal swing is seen first) -- using the whole history including future weeks would
    # leak information about weeks not yet observed at forecast time, the same mistake the
    # wastewater signal had and the leak test is built to catch.
    wk = wk.sort_values(["location", "week"])
    g = wk.groupby("location")
    temp_mean = g["temp_c"].transform(lambda s: s.shift(1).expanding(min_periods=52).mean())
    temp_std = g["temp_c"].transform(lambda s: s.shift(1).expanding(min_periods=52).std())
    hum_mean = g["humidity_pct"].transform(lambda s: s.shift(1).expanding(min_periods=52).mean())
    hum_std = g["humidity_pct"].transform(lambda s: s.shift(1).expanding(min_periods=52).std())
    temp_z = ((wk["temp_c"] - temp_mean) / temp_std.where(temp_std > 0)).clip(-4, 4)
    hum_z = ((wk["humidity_pct"] - hum_mean) / hum_std.where(hum_std > 0)).clip(-4, 4)
    wk["weather_level"] = (-temp_z - hum_z) / 2  # higher = colder and drier than that state's past norm
    name = "weather_state_week.csv"
    if signal == "ah":
        # absolute humidity (g/m3) from daily mean temperature and relative humidity, past-only z-score,
        # sign flipped: higher = drier air in absolute terms (Shaman & Kohn 2009)
        ah_mean = g["ah"].transform(lambda s: s.shift(1).expanding(min_periods=52).mean())
        ah_std = g["ah"].transform(lambda s: s.shift(1).expanding(min_periods=52).std())
        wk["weather_level"] = -((wk["ah"] - ah_mean) / ah_std.where(ah_std > 0)).clip(-4, 4)
        name = "weather_ah_state_week.csv"
    wk = wk.replace([np.inf, -np.inf], np.nan).dropna(subset=["weather_level"])

    out = wk.rename(columns={"week": "date"})[["date", "location", "weather_level"]]
    out.to_csv(ROOT / "data/clean" / name, index=False)
    print(f"Wrote data/clean/{name}: {len(out):,} rows, "
          f"{out['location'].nunique()} states, {out['date'].min().date()} to {out['date'].max().date()}")


if __name__ == "__main__":
    import sys
    main("ah" if "--ah" in sys.argv else "index")
