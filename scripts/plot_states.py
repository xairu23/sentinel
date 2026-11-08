"""Plot weekly flu hospital admissions for a few example states, to see what a flu season looks like.

Run:  python scripts/plot_states.py   (after ingest.py)
Output: figures/example_states.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "clean" / "hosp_flu.csv", dtype={"location": str}, parse_dates=["date"])

STATES = ["US", "06", "48", "36", "12", "17"]  # US total, California, Texas, New York, Florida, Illinois
fig, axes = plt.subplots(len(STATES), 1, figsize=(10, 2.2 * len(STATES)), sharex=True)
for ax, loc in zip(axes, STATES):
    s = df[df.location == loc]
    ax.plot(s.date, s.value, lw=1.4)
    ax.set_title(s.location_name.iloc[0], loc="left", fontsize=10)
    ax.set_ylabel("admissions/wk", fontsize=8)
    ax.grid(alpha=0.3)
fig.suptitle("Weekly lab-confirmed flu hospital admissions (CDC FluSight target data)", fontsize=11)
fig.tight_layout()
out = ROOT / "figures" / "example_states.png"
fig.savefig(out, dpi=130)
print(f"Saved {out}")
