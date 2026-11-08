"""Quick look: does the wastewater signal move BEFORE hospital admissions?
For each state, correlate log hospital admissions at week t with wastewater at week t-k (k = 0..4).
Output: printed median correlation per lag + figures/wastewater_vs_hosp.png"""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
h = pd.read_csv(ROOT/"data/clean/hosp_flu.csv", dtype={"location": str}, parse_dates=["date"])
w = pd.read_csv(ROOT/"data/clean/wastewater_state_week.csv", dtype={"location": str}, parse_dates=["date"])
d = h.merge(w, on=["date", "location"], how="inner")
d["log_h"] = np.log1p(d["value"])
rows = []
for loc, g in d.groupby("location"):
    g = g.set_index("date").sort_index()
    if len(g) < 80: continue
    for k in range(0, 5):
        rows.append((loc, k, g["log_h"].corr(g["ww_level"].shift(k))))
r = pd.DataFrame(rows, columns=["loc", "lag", "corr"])
print("States analysed:", r["loc"].nunique())
print(r.groupby("lag")["corr"].median().round(3).to_string())
r = r.dropna(); best = r.loc[r.groupby("loc")["corr"].idxmax()]["lag"].value_counts().sort_index()
print("Best lag (weeks wastewater leads) count of states:\n", best.to_string())
fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
for ax, loc in zip(axes, ["06", "48", "36"]):
    g = d[d.location == loc].sort_values("date")
    ax.plot(g.date, g.log_h, label="hospital admissions (log)", lw=1.4)
    ax2 = ax.twinx(); ax2.plot(g.date, g.ww_level, color="tab:orange", lw=1.2, label="wastewater (z)")
    ax.set_title(g.location_name.iloc[0], loc="left", fontsize=10); ax.grid(alpha=.3)
axes[0].legend(loc="upper left", fontsize=8); fig.tight_layout()
fig.savefig(ROOT/"figures/wastewater_vs_hosp.png", dpi=130)
