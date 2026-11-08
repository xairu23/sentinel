"""Build this week's forecast file in the exact CDC FluSight hub format.

Run from the repo root:
    python -m submission.build_submission                     # next Saturday as reference date
    python -m submission.build_submission --reference-date 2026-10-17

Steps:
 1. load clean hospital data, ER visits, wastewater and weather (scripts/ingest.py + the clean scripts)
 2. last = reference_date - 7 days (the latest week with reported admissions)
 3. run Sentinel v2 (models/engine.py): reporting-delay fix, signals with their real publication
    delays, 8 regressions (every combination of ER / wastewater / weather) averaged, then averaged
    with equal weights with two other model families: the trees trained on 10 extra seasons of
    history (models/gbq.py) and the quantile autoregression (models/qar.py)
 4. write submission/output/YYYY-MM-DD-Sentinel-Ens.csv (hub columns, 23 quantiles as whole numbers,
    FIPS codes), plus the optional trend target 'wk flu hosp rate change' (5 category probabilities,
    derived from the same quantiles: submission/rate_change.py)
"""
import argparse
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.wis import QUANTILES
from backtest.run_backtest import load_signals
from models import engine
from models.models import K_STEPS
from submission.rate_change import pmf as rc_pmf

ROOT = Path(__file__).resolve().parents[1]
TEAM_MODEL = "Sentinel-Ens"

# TEMPORARY early-season adjustment (added 2026-10-21, team decision; see backtest/round20.py).
# In the first live weeks our forecasts for small places ran ~10% high vs the top teams. Until the date
# below, places whose newest published week is under EARLY_MAX_LEVEL get their quantiles lowered by
# EARLY_SHIFT on the log scale (~5%). Replay: -2.5% on those rows in 2023-24 and 2024-25, +3% in 2025-26;
# about -0.1% overall. Remove after EARLY_UNTIL.
EARLY_SHIFT = 0.0  # set to 0.05 to switch on
EARLY_MAX_LEVEL = 100
EARLY_UNTIL = date(2026, 11, 30)


def next_saturday(today: date) -> date:
    return today + timedelta(days=(5 - today.weekday()) % 7)


def load_vintages() -> pd.DataFrame | None:
    """CDC's archive of every published version (downloaded by scripts/ingest.py). Needed for the
    reporting-delay correction; if it's missing the forecast still runs, just without that fix."""
    for p in (ROOT / "data/raw/time-series.csv", ROOT.parent / "FluSight-forecast-hub/target-data/time-series.csv"):
        if p.exists():
            t = pd.read_csv(p, dtype={"location": str}, parse_dates=["as_of", "target_end_date"])
            t = t[t["target"] == "wk inc flu hosp"]
            return t.rename(columns={"target_end_date": "date", "observation": "value"})
    print("WARNING: no published-versions archive found; skipping the reporting-delay correction")
    return None


def build(reference_date: date, cfg: engine.Config = engine.FINAL, out_dir: Path | None = None,
          rate_change: bool = True) -> pd.DataFrame:
    """This week's forecast with Sentinel (models/engine.py), the same engine the replay tests:
    hospital data as published now, ER visits for the same newest week, wastewater one week behind,
    weather, the reporting-delay fix, the 8 regressions averaged, then the equal-weight average with
    the tree model and the quantile autoregression (v3)."""
    ref = pd.Timestamp(reference_date)
    last = ref - pd.Timedelta(weeks=1)
    h = pd.read_csv(ROOT / "data/clean/hosp_flu.csv", dtype={"location": str}, parse_dates=["date"])
    h = h[["date", "location", "value"]]
    have = h.loc[h["date"] == last, "location"].unique()
    if len(have) == 0:
        raise SystemExit(f"No hospital data for week ending {last.date()} yet - run ingest after the hub update.")
    sig = load_signals()
    vint = load_vintages()
    lags = engine.lag_table(vint) if vint is not None else None
    as_of = vint["as_of"].max() + pd.Timedelta(days=1) if vint is not None else ref  # after every archived version
    pop = pd.read_csv(ROOT / "data/clean/locations.csv", dtype={"location": str}).set_index("location")["population"]
    hist = engine.load_history(cfg) if cfg.hist else None
    preds = engine.forecast(h, last, as_of, sig, lags, pop, cfg, states_only=False, hist=hist)
    if cfg.er_anchor:
        # v3.3: ER-visit anchor (round 26c); ER flu share exactly as downloaded this week
        ed_raw = pd.read_csv(ROOT / "data/clean/ed_flu.csv", dtype={"location": str}, parse_dates=["date"])
        if ed_raw["date"].max() < last:
            print(f"WARNING: ER file ends {ed_raw['date'].max().date()}, before {last.date()}; ER anchor skipped")
        else:
            pre = dict(preds)
            preds, used = engine.er_anchor(preds, h, ed_raw, last, *cfg.er_anchor)
            # shadow copy WITHOUT the anchor, for the weekly live check of round 26c's monitoring rule
            sh = ROOT / "submission/shadow/no_er_anchor"; sh.mkdir(parents=True, exist_ok=True)
            pd.DataFrame([{"reference_date": ref.date().isoformat(), "location": loc, "horizon": k - 1,
                           "output_type_id": f"{tau:g}", "value": int(round(float(v))), "anchored": loc in used}
                          for (loc, k), q in pre.items() for tau, v in zip(QUANTILES, q)]
                         ).to_csv(sh / f"{ref.date().isoformat()}-{TEAM_MODEL}.csv", index=False)
            print(f"ER anchor applied to {len(used)} places: " +
                  ", ".join(f"{k} {v:+.2f}" for k, v in sorted(used.items())))

    latest = h[h["date"] == last].set_index("location")["value"]
    if EARLY_SHIFT and ref.date() <= EARLY_UNTIL:
        n = 0
        for (loc, k), q in list(preds.items()):
            if loc != "US" and pd.notna(latest.get(loc)) and latest[loc] < EARLY_MAX_LEVEL:
                preds[(loc, k)] = np.expm1(np.log1p(np.maximum(np.asarray(q, float), 0)) - EARLY_SHIFT); n += 1
        print(f"Temporary early-season shift applied to {n} place-horizons (until {EARLY_UNTIL})")
    rows, trend = [], []
    for loc in sorted(have):
        for k in K_STEPS:
            q = preds.get((loc, k))
            if q is None:
                continue
            hz = k - 1
            end = (ref + pd.Timedelta(weeks=hz)).date().isoformat()
            # whole numbers (hub rule since 2025-26); rounding keeps the quantiles non-decreasing
            qi = [int(round(float(v))) for v in q]
            for tau, v in zip(QUANTILES, qi):
                rows.append({"reference_date": ref.date().isoformat(), "target": "wk inc flu hosp",
                             "horizon": hz, "target_end_date": end, "location": loc, "output_type": "quantile",
                             "output_type_id": f"{tau:g}", "value": v})
            if rate_change and loc in pop.index and pd.notna(latest.get(loc)):
                # optional hub target: probabilities of the 5 trend categories, from the same quantiles
                for cat, pv in rc_pmf(np.array(qi, float), float(latest[loc]), float(pop[loc]), hz).items():
                    trend.append({"reference_date": ref.date().isoformat(), "target": "wk flu hosp rate change",
                                  "horizon": hz, "target_end_date": end, "location": loc, "output_type": "pmf",
                                  "output_type_id": cat, "value": float(pv)})
    missing = sorted({loc for loc in have for k in K_STEPS if preds.get((loc, k)) is None})
    if missing or len(have) < len(pop):
        print(f"WARNING: no forecast for {', '.join(missing) or '-'}; "
              f"{len(have)} of {len(pop)} places have data for {last.date()}")
    sub = pd.DataFrame(rows + trend)
    out_dir = Path(out_dir).resolve() if out_dir else ROOT / "submission/output"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{ref.date().isoformat()}-{TEAM_MODEL}.csv"
    sub.to_csv(path, index=False)
    print(f"Wrote {path}: {sub['location'].nunique()} locations, horizons {sorted(sub['horizon'].unique())}, "
          f"{len(sub):,} rows ({len(rows):,} quantiles, {len(trend):,} trend probabilities; data through {last.date()})")
    return sub


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference-date", type=date.fromisoformat, default=next_saturday(date.today()))
    ap.add_argument("--model", choices=["final", "v35", "v34", "v33", "v32", "v31", "v3", "v2"], default="final",
                    help="final = what is submitted (engine.FINAL, now v3.6); v35, v34, v33, v32, v31, v3 or v2 for comparison")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--no-rate-change", action="store_true", help="only the required quantile target")
    a = ap.parse_args()
    cfg = {"final": engine.FINAL, "v35": engine.V35, "v34": engine.V34, "v33": engine.V33, "v32": engine.V32, "v31": engine.V31, "v3": engine.V3, "v2": engine.V2}[a.model]
    build(a.reference_date, cfg, a.out_dir, not a.no_rate_change)
