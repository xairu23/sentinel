"""Weekly live check of the ER anchor (round 26c monitoring rule): for every submitted week that has a shadow
file without the anchor (submission/shadow/no_er_anchor/), score both versions on the anchored places with the
newest hospital data (data/raw/time-series.csv, latest version), log(x+1) WIS.
    python -m backtest.score_er_anchor
Switch the anchor off (engine.FINAL = V32 or er_anchor=()) if it is worse over the first 4 scored weeks in
which it was active."""
import glob
from pathlib import Path
import numpy as np, pandas as pd
from backtest.wis import QUANTILES as QS
ROOT = Path(__file__).resolve().parents[1]
t = pd.read_csv(ROOT / "data/raw/time-series.csv", dtype={"location": str}, parse_dates=["as_of", "target_end_date"])
t = t[t.target == "wk inc flu hosp"]; truth = t[t.as_of == t.as_of.max()].set_index(["target_end_date", "location"]).observation
wis = lambda v, y: np.mean(2 * (((y < v) - QS) * (v - y)))
rows = []
for f in sorted(glob.glob(str(ROOT / "submission/shadow/no_er_anchor/*.csv"))):
    sh = pd.read_csv(f, dtype={"location": str}); ref = pd.Timestamp(sh.reference_date.iloc[0])
    sub = pd.read_csv(ROOT / "submission/output" / Path(f).name, dtype={"location": str})
    sub = sub[(sub.target == "wk inc flu hosp") & (sub.output_type == "quantile")]
    for (loc, hz), g in sh[sh.anchored].groupby(["location", "horizon"]):
        y = truth.get((ref + pd.Timedelta(weeks=int(hz)), loc))
        if y is None or np.isnan(y): continue
        s = sub[(sub.location == loc) & (sub.horizon == hz)].sort_values("output_type_id").value.values.astype(float)
        a = g.sort_values("output_type_id").value.values.astype(float)
        rows.append((ref.date(), loc, hz, wis(np.log1p(s), np.log1p(y)), wis(np.log1p(a), np.log1p(y))))
R = pd.DataFrame(rows, columns=["ref", "loc", "h", "with_anchor", "without"])
if R.empty:
    print("no scored anchored forecasts yet")
else:
    by = R.groupby("ref")[["with_anchor", "without"]].sum(); by["change"] = by.with_anchor / by.without - 1
    print(by.round(4).to_string()); print(f"total: {R.with_anchor.sum()/R.without.sum()-1:+.2%} over {R.ref.nunique()} weeks, {len(R)} forecasts")
