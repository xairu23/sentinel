"""Long flu history for training: turns the raw CDC files in data/history/ into one table.

Sources (public CDC data, copied from the UMass Reich Lab "flusion" repository, MIT licence,
https://github.com/reichlab/flusion -- see data/history/LICENSE-flusion-MIT.txt):
  - FluSurv-NET: lab-confirmed influenza hospitalisation rate per 100k, weekly, 14 sites + network,
    seasons 2009-10 to 2019-20 (old-flusurv-rates.csv, "Overall" age group)
  - ILINet: share of outpatient visits for influenza-like illness, by state (ilinet_state.csv)
  - WHO/NREVSS clinical labs: % of flu tests positive, by state (who-nrevss.csv)
    ILI x % positive = "ILI+", the standard proxy for actual flu activity.

Only data BEFORE 1 July 2022 is kept (more than a year before the first week Sentinel is ever
tested on), and the 2009-10 pandemic and the two COVID seasons (2020-21, 2021-22, when flu all but
disappeared) are dropped.

Output: data/history/sentinel_history.csv  (source, location, epiweek, v1, v2)
  flusurv rows: v1 = weekly rate per 100k
  iliplus rows: v1 = ILI %, v2 = % positive
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
H = ROOT / "data/history"
DROP_SEASONS = {"2009-10", "2009/10", "2020-21", "2020/21", "2021-22", "2021/22"}

FLUSURV_CODES = {"California": "CA", "Colorado": "CO", "Connecticut": "CT", "Georgia": "GA",
                 "Maryland": "MD", "Michigan": "MI", "Minnesota": "MN", "New Mexico": "NM",
                 "New York - Albany": "NY_albany", "New York - Rochester": "NY_rochester",
                 "Ohio": "OH", "Oregon": "OR", "Tennessee": "TN", "Utah": "UT"}


def main() -> None:
    loc = pd.read_csv(ROOT / "data/clean/locations.csv", dtype={"location": str})
    name_to_abbr = dict(zip(loc["location_name"], loc["abbreviation"]))

    f = pd.read_csv(H / "old-flusurv-rates.csv", encoding="ISO-8859-1")
    f = f[(f["age_label"] == "Overall") & f["region"].isin(FLUSURV_CODES) & ~f["sea_label"].isin(DROP_SEASONS)]
    fs = pd.DataFrame({"source": "flusurv", "location": f["region"].map(FLUSURV_CODES),
                       "epiweek": f["epiweek"].astype(int), "v1": f["weeklyrate"], "v2": np.nan})

    i = pd.read_csv(H / "ilinet_state.csv")
    i = i[~i["season"].isin(DROP_SEASONS)]
    i = i.assign(epiweek=i["year"] * 100 + i["week"])
    w = pd.read_csv(H / "who-nrevss.csv")
    w = w[w["region_type"] == "States"].assign(epiweek=lambda d: d["year"] * 100 + d["week"])
    m = i.merge(w[["region", "epiweek", "percent_positive"]], on=["region", "epiweek"], how="inner")
    m["abbr"] = m["region"].map(name_to_abbr)
    m = m.dropna(subset=["abbr", "unweighted_ili", "percent_positive"])
    ip = pd.DataFrame({"source": "iliplus", "location": m["abbr"].str.lower(),
                       "epiweek": m["epiweek"].astype(int), "v1": m["unweighted_ili"],
                       "v2": m["percent_positive"]})

    out = pd.concat([fs, ip], ignore_index=True)
    out = out[out["epiweek"] < 202226]  # before July 2022
    out.to_csv(H / "sentinel_history.csv", index=False)
    print(out.groupby("source").agg(rows=("v1", "size"), locations=("location", "nunique"),
                                     first=("epiweek", "min"), last=("epiweek", "max")))


if __name__ == "__main__":
    main()
