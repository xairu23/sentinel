"""Charts for the README, redrawn automatically every week by the autopilot.

1. figures/latest_forecast.png  - this week's forecast (fan chart) for the US and 5 large states
2. figures/backtest_replay.png  - how the 1-week-ahead replay forecasts tracked reality in past seasons

Run from the repo root:  python scripts/plot_forecast.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PLACES = [("US", "United States"), ("06", "California"), ("48", "Texas"), ("12", "Florida"),
          ("36", "New York"), ("17", "Illinois")]
BLUE, FAN = "#1f5fa8", "#1f5fa8"


def latest_forecast():
    files = (sorted((ROOT / "submission/output").glob("*-Sentinel-Ens.csv"))
             or sorted((ROOT / "submission/sealed").glob("*-Sentinel-Ens.csv")))
    if not files:
        print("no forecast file yet")
        return
    f = pd.read_csv(files[-1], dtype={"location": str, "output_type_id": str}, parse_dates=["target_end_date"])
    obs = pd.read_csv(ROOT / "data/clean/hosp_flu.csv", dtype={"location": str}, parse_dates=["date"])
    ref = f["reference_date"].iloc[0]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    for ax, (loc, name) in zip(axes.flat, PLACES):
        o = obs[(obs.location == loc)].sort_values("date").tail(26)
        g = f[f.location == loc].pivot_table(index="target_end_date", columns="output_type_id", values="value")
        ax.plot(o.date, o.value, color="black", lw=1.5, marker="o", ms=2.5, label="reported")
        if not g.empty:
            ax.fill_between(g.index, g["0.025"], g["0.975"], color=FAN, alpha=0.15, label="95% range")
            ax.fill_between(g.index, g["0.25"], g["0.75"], color=FAN, alpha=0.35, label="50% range")
            ax.plot(g.index, g["0.5"], color=BLUE, lw=2, marker="o", ms=3, label="Sentinel median")
        ax.set_title(name, loc="left", fontsize=10)
        ax.grid(alpha=0.3)
        ax.tick_params(axis="x", labelsize=7, rotation=30)
    axes.flat[0].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"Sentinel forecast, reference date {ref}: weekly flu hospital admissions", fontsize=12)
    fig.tight_layout()
    fig.savefig(ROOT / "figures/latest_forecast.png", dpi=120)
    print("saved figures/latest_forecast.png")


def backtest_replay():
    p = next((f for f in (ROOT / "backtest/results_v3.csv", ROOT / "backtest/results_v2.csv") if f.exists()), None)
    if p is None:
        return
    r = pd.read_csv(p, dtype={"location": str}, parse_dates=["reference_date"])
    # horizon 0 = the week right after the latest data ("1 week ahead")
    r = r[(r.model == "ensemble") & (r.horizon == 0)]
    r["target_date"] = r["reference_date"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    for ax, (loc, name) in zip(axes.flat, PLACES[1:] + [("42", "Pennsylvania")]):
        g = r[r.location == loc].sort_values("target_date")
        for season, s in g.groupby("season"):
            ax.plot(s.target_date, s.observed, color="black", lw=1.4)
            ax.plot(s.target_date, s["median"], color=BLUE, lw=1.2, ls="--")
        ax.set_title(name, loc="left", fontsize=10)
        ax.grid(alpha=0.3)
        ax.tick_params(axis="x", labelsize=7, rotation=30)
    axes.flat[0].plot([], [], color="black", label="what really happened")
    axes.flat[0].plot([], [], color=BLUE, ls="--", label="Sentinel v3, 1 week ahead (median, replay)")
    axes.flat[0].legend(fontsize=7, loc="upper left")
    fig.suptitle("Replay of 3 past seasons with data as published each week: forecast 1 week after the latest data",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(ROOT / "figures/backtest_replay.png", dpi=120)
    print("saved figures/backtest_replay.png")


if __name__ == "__main__":
    latest_forecast()
    backtest_replay()
