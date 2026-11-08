"""Practice on the past: forecast every week of three past flu seasons as if it were that week, then score.

Two modes:
  revised  (default)  uses today's corrected hospital numbers for the past (easier, optimistic).
  realtime            uses, for every past week, the hospital data EXACTLY as published that week
                      (the CDC hub keeps every weekly version in target-data/time-series.csv, column as_of).
                      This is what a team really had when it forecast live, so it is the fair test.

Run from the repo root:
  python -m backtest.run_backtest                                   # revised mode
  python -m backtest.run_backtest --mode realtime --hub ../FluSight-forecast-hub
Outputs: backtest/results_scores[_realtime].csv, backtest/summary[_realtime].md

Both modes score against the final corrected numbers, exactly like the CDC evaluates teams.
Wastewater has no archived versions, so it is used with a 1-week delay and standardised on past samples only.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.wis import QUANTILES, covered, wis
from models.models import K_STEPS, MODELS, baseline, ensemble, load_neighbor_pairs, make_features, run_all

ROOT = Path(__file__).resolve().parents[1]
WW_DELAY = 1  # weeks of reporting delay assumed for wastewater (conservative)
ED_DELAY = 1  # same conservative delay for ER-visit data (no archived versions before Dec 2025)
LAB_DELAY = 1  # same conservative delay for clinical-lab positivity (no archived versions)
WEATHER_DELAY = 0  # weather is published same-day with no revision, so no delay needed
SEASONS = {  # forecast reference dates covered (Saturdays), roughly the hub's season windows
    "2023-24": ("2023-10-14", "2024-04-27"),
    "2024-25": ("2024-11-23", "2025-05-24"),  # starts after the mid-2024 national reporting gap
    "2025-26": ("2025-10-11", "2026-05-23"),
}
N_BOOT = 2000


def load_signals() -> dict:
    w = pd.read_csv(ROOT / "data/clean/wastewater_state_week.csv", dtype={"location": str}, parse_dates=["date"])
    # SENTINEL_ED_FILE: an alternative ER-visit file (e.g. an older download, to reproduce old caches)
    e = pd.read_csv(__import__("os").environ.get("SENTINEL_ED_FILE", str(ROOT / "data/clean/ed_flu.csv")),
                    dtype={"location": str}, parse_dates=["date"])
    e["ed_level"] = np.log(e["value"] + 1e-4)  # log share of ER visits that are flu
    l = pd.read_csv(ROOT / "data/clean/lab_state_week.csv", dtype={"location": str}, parse_dates=["date"])
    # SENTINEL_WEATHER_FILE: an alternative weather signal for experiments (e.g. weather_ah_state_week.csv)
    wf = __import__("os").environ.get("SENTINEL_WEATHER_FILE", "weather_state_week.csv")
    we = pd.read_csv(ROOT / "data/clean" / wf, dtype={"location": str}, parse_dates=["date"])
    return {"ww": w[["date", "location", "ww_level"]], "ed": e[["date", "location", "ed_level"]],
            "lab": l[["date", "location", "lab_level"]], "weather": we[["date", "location", "weather_level"]]}


def attach_signals(hosp: pd.DataFrame, sig: dict, keep_ww_states: bool = True) -> pd.DataFrame:
    """Add wastewater, ER-visit and lab-positivity signals to a hospital panel (date, location, value).
    Each signal is delayed (WW_DELAY / ED_DELAY / LAB_DELAY weeks) so a forecast only uses what would
    really have been published.
    keep_ww_states: keep only the 50 states (fixed list), so with/without comparisons use the same places."""
    p = hosp.copy()
    for name, delay in (("ww", WW_DELAY), ("ed", ED_DELAY), ("lab", LAB_DELAY), ("weather", WEATHER_DELAY)):
        col = f"{name}_level"
        p = p.merge(sig[name], on=["date", "location"], how="left").sort_values(["location", "date"])
        p[col] = p.groupby("location")[col].shift(delay)  # delay first, then fill short gaps
        p[col] = p.groupby("location")[col].transform(lambda s: s.interpolate(limit=2, limit_direction="forward"))
    if keep_ww_states:
        # A FIXED list decided in advance (the 50 states), NOT "states that ever have wastewater":
        # choosing places by whether they get data in the future would quietly use future information.
        p = p[~p["location"].isin(["US", "11", "72"])]  # drop the national total, DC and Puerto Rico
    return p


def final_hosp() -> pd.DataFrame:
    h = pd.read_csv(ROOT / "data/clean/hosp_flu.csv", dtype={"location": str}, parse_dates=["date"])
    return h[["date", "location", "value"]]


class Vintages:
    """Every weekly version of the hospital data, as published at the time."""

    def __init__(self, hub: Path):
        t = pd.read_csv(hub / "target-data/time-series.csv", dtype={"location": str},
                        parse_dates=["as_of", "target_end_date"])
        t = t[t["target"] == "wk inc flu hosp"]
        self.t = t.rename(columns={"target_end_date": "date", "observation": "value"})
        self.max_date = self.t.groupby("as_of")["date"].max()

    def as_published(self, last: pd.Timestamp) -> pd.DataFrame | None:
        """The FIRST published version whose newest week is `last` (what forecasters had that Wednesday)."""
        ok = self.max_date[self.max_date == last]
        if ok.empty:
            return None
        v = self.t[self.t["as_of"] == ok.index.min()]
        return v[["date", "location", "value"]]


def forecast_one_week(panel: pd.DataFrame, feat, last, truth, season, ref, rows):
    preds = run_all(feat, last)
    for loc, s in panel[panel["date"] <= last].groupby("location"):
        s = s.set_index("date")["value"].dropna()
        if len(s) < 30:
            continue
        for k in K_STEPS:
            y = truth.get((loc, last + pd.Timedelta(weeks=k)))
            if y is None or np.isnan(y):
                continue
            if any((loc, k) not in d for d in preds.values()):
                continue  # score only when every model produced a forecast (paired comparison)
            cands = {"baseline": baseline(s, k)}
            cands.update({name: d[(loc, k)] for name, d in preds.items()})
            cands["ensemble"] = ensemble(preds, loc, k)
            for name, q in cands.items():
                rows.append({"season": season, "reference_date": ref.date(), "location": loc,
                             "horizon": k - 1, "model": name, "observed": y,
                             "median": q[QUANTILES == 0.5][0], "wis": wis(q, y),
                             "cov50": covered(q, y, 0.5), "cov95": covered(q, y, 0.95)})


def main(mode: str, hub: Path):
    load_neighbor_pairs(ROOT / "data/clean/locations.csv")
    sig = load_signals()
    final = attach_signals(final_hosp(), sig)
    truth = final.set_index(["location", "date"])["value"]
    if mode == "revised":
        feat = make_features(final)
    else:
        vint = Vintages(hub)
    rows, skipped = [], 0
    for season, (start, end) in SEASONS.items():
        for ref in pd.date_range(start, end, freq="7D"):
            last = ref - pd.Timedelta(weeks=1)
            if mode == "revised":
                panel = final
            else:
                v = vint.as_published(last)
                if v is None:
                    skipped += 1
                    continue
                panel = attach_signals(v, sig)
                feat = make_features(panel)
            forecast_one_week(panel, feat, last, truth, season, ref, rows)
        print(f"season {season} done, rows so far: {len(rows):,}", flush=True)
    if skipped:
        print(f"weeks skipped (no published version for that week): {skipped}")
    res = pd.DataFrame(rows)
    suffix = "" if mode == "revised" else "_realtime"
    res.to_csv(ROOT / f"backtest/results_scores{suffix}.csv", index=False)
    write_summary(res, mode, suffix, skipped)


def rel_wis(res: pd.DataFrame) -> pd.Series:
    m = res.groupby("model")["wis"].mean()
    return (m / m["baseline"]).round(3)


def bootstrap_change(piv: pd.DataFrame, a: str, b: str, seed: int = 0) -> tuple[float, float, float]:
    """% change in mean WIS of model b vs model a, with a 95% confidence interval.
    Resamples whole forecast weeks (all locations/horizons of a week together), because forecasts
    made the same week share the same epidemic situation and are not independent."""
    rng = np.random.default_rng(seed)
    by_week = piv.groupby(level="reference_date")[[a, b]].sum()
    weeks = by_week.index.values
    point = (piv[b].mean() / piv[a].mean() - 1) * 100
    boots = []
    for _ in range(N_BOOT):
        pick = rng.choice(weeks, size=len(weeks), replace=True)
        sa, sb = by_week.loc[pick, a].sum(), by_week.loc[pick, b].sum()
        boots.append((sb / sa - 1) * 100)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi


def write_summary(res: pd.DataFrame, mode: str, suffix: str, skipped: int):
    title = "revised data (optimistic)" if mode == "revised" else "real-time data, as published each week (fair)"
    lines = [f"# Backtest summary: {title}", "",
             "Relative WIS = model's mean WIS / Sentinel baseline's mean WIS (below 1 = better).",
             "Same locations, dates and horizons for every model (paired).", ""]
    overall = pd.DataFrame({"relative WIS": rel_wis(res),
                            "50% coverage": res.groupby("model")["cov50"].mean().round(3),
                            "95% coverage": res.groupby("model")["cov95"].mean().round(3)})
    lines += ["## All three seasons", "", overall.to_markdown(), ""]
    by = res.groupby("season").apply(rel_wis, include_groups=False)
    lines += ["## Relative WIS by season", "", by.to_markdown(), ""]
    byh = res.groupby("horizon").apply(rel_wis, include_groups=False)
    lines += ["## Relative WIS by horizon (weeks)", "", byh.to_markdown(), ""]
    piv = res.pivot_table(index=["season", "reference_date", "location", "horizon"],
                          columns="model", values="wis")
    lines += ["## Research question: does wastewater help?", "",
              "Each line compares two models that differ ONLY by the wastewater signal.",
              "Change in mean WIS with 95% confidence interval (bootstrap over forecast weeks).", ""]
    for a, b, label in (("growth", "growth_ww", "hospital trends only  ->  + wastewater"),
                        ("growth_ed", "growth_ed_ww", "hospital trends + ER visits  ->  + wastewater")):
        pt, lo, hi = bootstrap_change(piv, a, b)
        wins = (piv[b] < piv[a]).mean() * 100
        verdict = "no clear effect" if lo < 0 < hi else ("helps" if hi < 0 else "hurts")
        lines.append(f"- {label}: {pt:+.1f}% (95% CI {lo:+.1f}% to {hi:+.1f}%), better in {wins:.0f}% of "
                     f"{len(piv):,} forecasts -> **{verdict}**")
    lines += ["", "## Experiment: neighbouring states", ""]
    for a, b, label in (("growth", "growth_nb", "hospital trends  ->  + neighbours"),
                        ("growth_ed_ww", "growth_ed_ww_nb", "hospital + ER + wastewater  ->  + neighbours")):
        pt, lo, hi = bootstrap_change(piv, a, b)
        wins = (piv[b] < piv[a]).mean() * 100
        verdict = "no clear effect" if lo < 0 < hi else ("helps" if hi < 0 else "hurts")
        lines.append(f"- {label}: {pt:+.1f}% (95% CI {lo:+.1f}% to {hi:+.1f}%), better in {wins:.0f}% -> **{verdict}**")
    lines += ["", "## Experiment: lab-test positivity", ""]
    for a, b, label in (("growth", "growth_lab", "hospital trends  ->  + lab positivity"),
                        ("growth_ed_ww", "growth_ed_ww_lab", "hospital + ER + wastewater  ->  + lab positivity"),
                        ("growth_ed_ww_nb", "growth_ed_ww_nb_lab", "hospital + ER + wastewater + neighbours  ->  + lab positivity")):
        pt, lo, hi = bootstrap_change(piv, a, b)
        wins = (piv[b] < piv[a]).mean() * 100
        verdict = "no clear effect" if lo < 0 < hi else ("helps" if hi < 0 else "hurts")
        lines.append(f"- {label}: {pt:+.1f}% (95% CI {lo:+.1f}% to {hi:+.1f}%), better in {wins:.0f}% -> **{verdict}**")
    lines += ["", "## Experiment: weather (cold/dry conditions)", ""]
    for a, b, label in (("growth", "growth_weather", "hospital trends  ->  + weather"),
                        ("growth_ed_ww", "growth_ed_ww_weather", "hospital + ER + wastewater  ->  + weather")):
        pt, lo, hi = bootstrap_change(piv, a, b)
        wins = (piv[b] < piv[a]).mean() * 100
        verdict = "no clear effect" if lo < 0 < hi else ("helps" if hi < 0 else "hurts")
        lines.append(f"- {label}: {pt:+.1f}% (95% CI {lo:+.1f}% to {hi:+.1f}%), better in {wins:.0f}% -> **{verdict}**")
    pt, lo, hi = bootstrap_change(piv, "growth", "growth_ed")
    lines += ["", "## Side question: do ER visits help?", "",
              f"- hospital trends only  ->  + ER visits: {pt:+.1f}% (95% CI {lo:+.1f}% to {hi:+.1f}%)"]
    ept, elo, ehi = bootstrap_change(piv, "baseline", "ensemble")
    lines += ["", "## Sentinel ensemble vs Sentinel baseline", "",
              f"- Change in mean WIS: {ept:+.1f}% (95% CI {elo:+.1f}% to {ehi:+.1f}%)", "",
              "## Limitations", ""]
    if mode == "revised":
        lines.append("- Uses revised (current) data, not the data available in real time: optimistic.")
    else:
        lines.append("- Hospital data exactly as first published each week; scored against final data, like the CDC.")
        if skipped:
            lines.append(f"- {skipped} forecast weeks skipped because no published version existed for them.")
    lines += ["- Wastewater has no archived versions: used with a 1-week delay, standardised on past samples only.",
              "- ER-visit data archived only from Dec 2025: used with a 1-week delay throughout (conservative).",
              "- Model features were chosen while looking at these same seasons; the live 2026-27 season is the true test."]
    (ROOT / f"backtest/summary{suffix}.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["revised", "realtime"], default="revised")
    ap.add_argument("--hub", type=Path, default=ROOT.parent / "FluSight-forecast-hub")
    a = ap.parse_args()
    main(a.mode, a.hub)
