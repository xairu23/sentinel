"""v3 experiments, run by the rules in backtest/V3_PROTOCOL.md (leave-one-season-out).

All three past seasons are replayed from caches made with the data as published each week:
  mem_*   the 8 regression members (backtest/member_cache.py), ER visits as really published
  gbq6_*  the tree model, 6 bags (backtest/exp_gbq.py)
  qar_*   the quantile autoregression (models/qar.py), made here with --produce-qar

    python -m backtest.exp_v3 --produce-qar      # slow (~1-2 min per week), resumable
    python -m backtest.exp_v3                    # all experiments, seconds
"""
from __future__ import annotations

import argparse
import itertools
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from backtest.harness import EXCLUDE
from backtest.member_cache import CACHE, load, mean_members
from backtest.run_backtest import final_hosp
from backtest.wis import QUANTILES

ROOT = Path(__file__).resolve().parents[1]
SEASONS = ("2023-24", "2024-25", "2025-26")
Q50, Q25, Q75 = (np.where(np.isclose(QUANTILES, x))[0][0] for x in (0.5, 0.25, 0.75))
Q025, Q975 = (np.where(np.isclose(QUANTILES, x))[0][0] for x in (0.025, 0.975))


# ---------------------------------------------------------------------------------------------
# production of the QAR cache
def family_fn(fam: str):
    """Forecast function of a model family by cache name: f(hosp as published, last, pop) -> forecasts."""
    if fam == "qar":
        from models.qar import qar_forecast
        return qar_forecast
    if fam == "qarw":  # round 3, idea 12: hospital rows weighted 3
        from dataclasses import replace
        from models.qar import QAR, qar_forecast
        return lambda h, last, pop: qar_forecast(h, last, pop, replace(QAR, nhsn_weight=3.0))
    if fam == "gbqsp":  # round 4, idea 18: trees with neighbouring states
        from dataclasses import replace
        from models.gbq import GBQ, gbq_forecast
        return lambda h, last, pop: gbq_forecast(h, last, pop, replace(GBQ, bags=6, spatial=True))
    if fam == "gbqx":  # idea 6: trees with weeks from Christmas
        from dataclasses import replace
        from models.gbq import GBQ, gbq_forecast
        return lambda h, last, pop: gbq_forecast(h, last, pop, replace(GBQ, bags=6, xmas=True))
    if fam in ("qarall", "qarsw"):  # ideas 20a / 20b: QAR trained on all in-season weeks
        from dataclasses import replace
        from models.qar import QAR, qar_forecast
        cfg = replace(QAR, window="in_season", season_week_term=(fam == "qarsw"))
        return lambda h, last, pop: qar_forecast(h, last, pop, cfg)
    if fam == "ar6":  # round 3, idea 11
        from models.ar6 import ar6_forecast
        return ar6_forecast
    raise ValueError(fam)


def produce_qar(tag: str = "qar"):
    from backtest.harness import Replay
    fn = family_fn(tag)
    rep = Replay(SEASONS)
    pop = pd.read_csv(ROOT / "data/clean/locations.csv", dtype={"location": str}).set_index("location")["population"]
    for w in rep.weeks:
        f = CACHE / f"{tag}_{w.ref.date()}.pkl"
        if f.exists():
            continue
        hosp = w.v[~w.v["location"].isin(list(EXCLUDE))]
        f.write_bytes(pickle.dumps({"ref": w.ref, "season": w.season, "qar": fn(hosp, w.last, pop)}))
        print(w.ref.date(), "done", flush=True)


def produce_all(tag: str = "all53", families=("mem", "gbq", "qar")):
    """Same replay for ALL 53 places (50 states + DC + Puerto Rico + the US), exactly as the weekly
    forecast runs (states_only=False). One cache file per family and week ({tag}{family}_{date}.pkl),
    so one family can be redone without the others. The 50-state experiments above leave out the
    US, DC and PR; this is how they get scored."""
    from dataclasses import replace
    from backtest.harness import Replay
    from backtest.run_v2 import EdArchive
    from models import engine
    from models.gbq import GBQ, gbq_forecast
    from models.qar import qar_forecast
    rep = Replay(SEASONS)
    ed = EdArchive(ROOT.parent / "FluSight-forecast-hub", rep.sig["ed"])
    lags = engine.lag_table(rep.vint.t)
    pop = pd.read_csv(ROOT / "data/clean/locations.csv", dtype={"location": str}).set_index("location")["population"]
    for w in rep.weeks:
        published = w.v[w.v["date"] <= w.last]
        corrected = None
        for fam in families:
            f = CACHE / f"{tag}{fam}_{w.ref.date()}.pkl"
            if f.exists():
                continue
            if fam == "mem":
                sig = {"ww": rep.sig["ww"], "weather": rep.sig["weather"], "ed": ed.as_published(w.as_of, w.last)}
                out = engine.member_forecasts(w.v, w.last, w.as_of, sig, lags, pop, engine.LINEAR, states_only=False)
            elif fam == "gbq":
                out = gbq_forecast(published, w.last, pop, replace(GBQ, bags=6))
            elif fam == "qar":
                out = qar_forecast(published, w.last, pop)
            elif fam == "qared":  # round 33: QAR + ER visits (today's Wednesday schedule), data as published
                from models.qar import QAR
                out = qar_forecast(published, w.last, pop, replace(QAR, ed=True), ed=ed.as_published_today(w.as_of, w.last))
            elif fam.endswith("bf"):
                # round 5, idea 19: the family gets the reporting-delay-corrected data, like the regressions
                if corrected is None:
                    corrected = engine.backfill_correct(published, w.last, w.as_of, lags, engine.LINEAR)
                if fam == "gbqbf":
                    out = gbq_forecast(corrected, w.last, pop, replace(GBQ, bags=6))
                elif fam == "gbqwbf":  # + idea 12 for the trees: hospital rows weighted 3
                    out = gbq_forecast(corrected, w.last, pop, replace(GBQ, bags=6, nhsn_weight=3.0))
                elif fam == "gbqspbf":  # + idea 18: neighbouring states
                    out = gbq_forecast(corrected, w.last, pop, replace(GBQ, bags=6, spatial=True))
                elif fam == "gbqspnbf":  # round 31: neighbours + season position + national features
                    out = gbq_forecast(corrected, w.last, pop, replace(GBQ, bags=6, spatial=True, seasonpos=True, national=True))
                elif fam == "gbqedbf":  # round 32: v3.5 trees + ER visits (today's Wednesday schedule)
                    out = gbq_forecast(corrected, w.last, pop, replace(GBQ, bags=6, spatial=True, seasonpos=True,
                                       national=True, ed=True), ed=ed.as_published_today(w.as_of, w.last))
                elif fam == "gbqxbf":  # + idea 6: weeks from Christmas
                    out = gbq_forecast(corrected, w.last, pop, replace(GBQ, bags=6, xmas=True))
                else:
                    out = qar_forecast(corrected, w.last, pop)
            else:
                out = family_fn(fam)(published, w.last, pop)
            f.write_bytes(pickle.dumps({"ref": w.ref, "season": w.season, fam: out}))
        print(w.ref.date(), "done", flush=True)


# ---------------------------------------------------------------------------------------------
# scoring
class Book:
    """Truth + a fast scorer for {ref: {(loc, k): q}} forecasts on the 50 states."""

    def __init__(self):
        h = final_hosp()
        self.truth = h.set_index(["location", "date"])["value"]
        self.mem = load("mem")
        self.gbq = {r: d["gbq"] for r, d in load("gbq6").items()}
        self.qar = {r: d["qar"] for r, d in load("qar").items()}
        self.season = {r: d["season"] for r, d in self.mem.items()}
        self.refs = sorted(self.mem)
        # one fixed list of (ref, loc, k) scored, so every model is compared on identical forecasts
        rows = []
        for r in self.refs:
            last = r - pd.Timedelta(weeks=1)
            lin = mean_members(self.mem[r]["members"])
            for (loc, k) in lin:
                if loc in EXCLUDE:
                    continue
                y = self.truth.get((loc, last + pd.Timedelta(weeks=k)))
                if y is None or np.isnan(y):
                    continue
                rows.append((self.season[r], r, loc, k, y))
        self.keys = pd.DataFrame(rows, columns=["season", "ref", "location", "k", "y"])

    def score(self, preds: dict) -> pd.DataFrame:
        """preds: {ref: {(loc, k): q}} -> keys + wis (+ coverage, median)."""
        Q = np.vstack([preds[r][(l, k)] for r, l, k in zip(self.keys.ref, self.keys.location, self.keys.k)])
        y = self.keys["y"].values[:, None]
        pin = ((y <= Q).astype(float) - QUANTILES) * (Q - y)
        d = self.keys.copy()
        d["wis"] = 2.0 * pin.mean(axis=1)
        d["cov50"] = (Q[:, Q25] <= d["y"]) & (d["y"] <= Q[:, Q75])
        d["cov95"] = (Q[:, Q025] <= d["y"]) & (d["y"] <= Q[:, Q975])
        d["median"] = Q[:, Q50]
        return d


def blend(a: dict, b: dict, w: float) -> dict:
    return {key: np.maximum.accumulate(np.maximum((1 - w) * a[key] + w * b[key], 0)) if key in b else a[key]
            for key in a}


def quantile_average(comps: list, weights=None, how: str = "mean") -> dict:
    """Combine forecasts {(loc, k): q} of several model families quantile by quantile."""
    out = {}
    for key in comps[0]:
        qs = np.array([c[key] for c in comps if key in c])
        if how == "median":
            q = np.median(qs, axis=0)
        elif weights is None:
            q = np.mean(qs, axis=0)  # exactly what models/engine.py does for v3
        else:
            w = np.ones(len(qs)) if weights is None else np.array([wt for c, wt in zip(comps, weights) if key in c])
            q = (w[:, None] * qs).sum(0) / w.sum()
        out[key] = np.maximum.accumulate(np.maximum(q, 0))
    return out


def adaptive_weights(book: "Book", comps: dict, window: int = 10, min_weeks: int = 3, shrink: float = 0.5) -> dict:
    """Idea 8: per forecast week, weights from each family's WIS on its forecasts of the last
    `window` weeks whose target week is already published (scored against the data AS PUBLISHED
    that week). Returns {ref: {name: weight}}."""
    from backtest.run_backtest import Vintages
    V = Vintages(ROOT.parent / "FluSight-forecast-hub")
    names = list(comps)
    out = {}
    for r in book.refs:
        last = r - pd.Timedelta(weeks=1)
        as_of = V.max_date[V.max_date == last].index.min()
        pub = V.t[V.t["as_of"] == as_of].set_index(["location", "date"])["value"]
        tot = {n: 0.0 for n in names}
        weeks = set()
        for r2 in book.refs:
            if not (r - pd.Timedelta(weeks=window) <= r2 < r):
                continue
            last2 = r2 - pd.Timedelta(weeks=1)
            for (loc, k), q in comps[names[0]][r2].items():
                if loc in EXCLUDE or last2 + pd.Timedelta(weeks=k) > last:
                    continue
                y = pub.get((loc, last2 + pd.Timedelta(weeks=k)))
                if y is None or np.isnan(y) or any((loc, k) not in comps[n][r2] for n in names):
                    continue
                weeks.add(r2)
                for n in names:
                    qq = comps[n][r2][(loc, k)]
                    tot[n] += 2.0 * (((y <= qq).astype(float) - QUANTILES) * (qq - y)).mean()
        if len(weeks) < min_weeks or min(tot.values()) <= 0:
            out[r] = {n: 1 / len(names) for n in names}
        else:
            inv = np.array([1 / tot[n] for n in names])
            w = shrink / len(names) + (1 - shrink) * inv / inv.sum()
            out[r] = dict(zip(names, w))
    return out


def change_ci(base: pd.DataFrame, new: pd.DataFrame, n_boot: int = 2000, seed: int = 0):
    """% change of total WIS, 95% CI resampling whole forecast weeks (as in backtest/harness.py)."""
    p = base[["ref", "wis"]].assign(new=new["wis"].values)
    wk = p.groupby("ref")[["wis", "new"]].sum()
    A, B = wk["wis"].values, wk["new"].values
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(A), size=(n_boot, len(A)))
    boots = (B[idx].sum(1) / A[idx].sum(1) - 1) * 100
    return (B.sum() / A.sum() - 1) * 100, *np.percentile(boots, [2.5, 97.5])


def by_season(base: pd.DataFrame, new: pd.DataFrame) -> dict:
    out = {}
    for s in SEASONS:
        m = (base["season"] == s).values
        out[s] = (new["wis"].values[m].sum() / base["wis"].values[m].sum() - 1) * 100
    return out


# ---------------------------------------------------------------------------------------------
# leave-one-season-out choice of a parameter
def loso(book: Book, candidates: dict, label: str, base: pd.DataFrame, per_key=None):
    """candidates: {theta: scored df}. For each season, theta is chosen on the OTHER two seasons
    (lowest total WIS), and that season is scored with it. per_key: optional function splitting
    the choice by a column (e.g. horizon), so each horizon gets its own theta."""
    groups = [None] if per_key is None else sorted(book.keys[per_key].unique())
    out = base.copy()
    chosen = {}
    for s in SEASONS:
        for g in groups:
            sel_train = np.array(book.keys["season"] != s)
            sel_test = np.array(book.keys["season"] == s)
            if g is not None:
                gm = np.array(book.keys[per_key] == g)
                sel_train &= gm
                sel_test &= gm
            best = min(candidates, key=lambda t: candidates[t]["wis"].values[sel_train].sum())
            chosen[(s, g)] = best
            out.loc[sel_test, "wis"] = candidates[best]["wis"].values[sel_test]
            out.loc[sel_test, ["cov50", "cov95", "median"]] = candidates[best].loc[sel_test, ["cov50", "cov95", "median"]].values
    return out, chosen


def report(name: str, base: pd.DataFrame, new: pd.DataFrame, chosen=None) -> dict:
    pt, lo, hi = change_ci(base, new)
    bs = by_season(base, new)
    keep = hi < 0 and all(v <= 0 for v in bs.values())
    line = (f"{name:44s} {pt:+6.1f}% ({lo:+.1f} to {hi:+.1f})  "
            + "  ".join(f"{s}: {v:+.1f}%" for s, v in bs.items()) + ("   KEEP" if keep else ""))
    print(line, flush=True)
    if chosen:
        print("      chosen:", chosen, flush=True)
    return {"idea": name, "change": pt, "lo": lo, "hi": hi, **bs, "keep": keep}


def main():
    book = Book()
    have_qar = len(book.qar) == len(book.refs)
    print(f"{len(book.refs)} weeks, {len(book.keys):,} forecasts (50 states); QAR cached for {len(book.qar)} weeks")
    lin = {r: mean_members(book.mem[r]["members"]) for r in book.refs}
    v2p = {r: blend(lin[r], book.gbq[r], 0.33) for r in book.refs}
    v2 = book.score(v2p)

    # sanity: this must reproduce the official v2 results file exactly
    off = pd.read_csv(ROOT / "backtest/results_v2.csv", dtype={"location": str}, parse_dates=["reference_date"])
    off = off[off["model"] == "ensemble"].rename(columns={"reference_date": "ref"})
    off["k"] = off["horizon"] + 1
    m = v2.merge(off[["ref", "location", "k", "wis"]], on=["ref", "location", "k"], suffixes=("", "_off"))
    print(f"v2 now vs the original v2 results file: {len(m):,} shared forecasts, max |WIS difference| "
          f"{np.abs(m['wis'] - m['wis_off']).max():.2e}; total WIS change "
          + ", ".join(f"{s}: {(m.wis[m.season == s].sum() / m.wis_off[m.season == s].sum() - 1) * 100:+.2f}%"
                      for s in SEASONS) + " (the week-53 calendar fix)")
    base_rel = {s: v2.loc[v2.season == s, "wis"].sum() for s in SEASONS}

    rows = []
    # idea 1: tree weight by horizon (grid 0..0.7), LOSO
    grid = [round(x, 1) for x in np.arange(0, 0.71, 0.1)]
    cands = {w: book.score({r: blend(lin[r], book.gbq[r], w) for r in book.refs}) for w in grid}
    new, ch = loso(book, cands, "tree weight", v2)
    rows.append(report("0. one tree weight for all horizons (LOSO)", v2, new, ch))
    new, ch = loso(book, cands, "tree weight", v2, per_key="k")
    rows.append(report("1. tree weight per horizon (LOSO)", v2, new, ch))

    # idea 2: median of the 8 regressions instead of the mean
    def med(members):
        out = {}
        for key in set().union(*members.values()):
            qs = [p[key] for p in members.values() if key in p]
            out[key] = np.maximum.accumulate(np.maximum(np.median(qs, axis=0), 0))
        return out
    lin_med = {r: med(book.mem[r]["members"]) for r in book.refs}
    rows.append(report("2. median of regressions (tree 0.33)", v2,
                       book.score({r: blend(lin_med[r], book.gbq[r], 0.33) for r in book.refs})))

    # idea 3: QAR as a third model, weight by LOSO (on top of v2's 0.33 trees)
    if have_qar:
        qa = book.score(book.qar)
        print(f"   QAR alone vs v2: {change_ci(v2, qa)[0]:+.1f}%  (cov95 {qa.cov95.mean():.2f}, cov50 {qa.cov50.mean():.2f})")
        cands = {w: book.score({r: blend(v2p[r], book.qar[r], w) for r in book.refs}) for w in grid}
        new, ch = loso(book, cands, "qar weight", v2)
        rows.append(report("3. + QAR, one weight (LOSO)", v2, new, ch))
        new, ch = loso(book, cands, "qar weight", v2, per_key="k")
        rows.append(report("3b. + QAR, weight per horizon (LOSO)", v2, new, ch))

    # idea 4: two training windows (2 and 3 years) averaged
    mem104 = load("mem104")
    if len(mem104) == len(book.refs):
        lin2 = {r: mean_members({**{("w104",) + c: p for c, p in mem104[r]["members"].items()},
                                 **book.mem[r]["members"]}) for r in book.refs}
        rows.append(report("4. 2- and 3-year windows averaged", v2,
                           book.score({r: blend(lin2[r], book.gbq[r], 0.33) for r in book.refs})))

    # idea 7 (disclosed, post-hoc): equal weights across the three families
    if have_qar:
        eq3 = {r: quantile_average([lin[r], book.gbq[r], book.qar[r]]) for r in book.refs}
        e3 = book.score(eq3)
        rows.append(report("7. equal weights: regressions/trees/QAR", v2, e3))
        rows.append(report("7b. same, median instead of mean", v2,
                           book.score({r: quantile_average([lin[r], book.gbq[r], book.qar[r]], how="median")
                                       for r in book.refs})))
        rows.append(report("7c. equal weights: regressions/trees only", v2,
                           book.score({r: quantile_average([lin[r], book.gbq[r]]) for r in book.refs})))
        # idea 8 (disclosed, post-hoc): weights adapting within the season, judged against idea 7
        comps = {"lin": lin, "gbq": book.gbq, "qar": book.qar}
        aw = adaptive_weights(book, comps)
        ad = book.score({r: quantile_average([lin[r], book.gbq[r], book.qar[r]],
                                             [aw[r]["lin"], aw[r]["gbq"], aw[r]["qar"]]) for r in book.refs})
        rows.append(report("8. adaptive weights (vs v2)", v2, ad))
        rows.append(report("8. adaptive weights (vs idea 7)", e3, ad))
        pickle.dump(aw, open(CACHE / "v3_adaptive_weights.pkl", "wb"))

        # idea 10 (pre-registered after the second review): one spread factor for v3's ranges, LOSO
        def spread(pred, c):
            return {key: np.maximum.accumulate(np.maximum(q[11] + c * (q - q[11]), 0)) for key, q in pred.items()}
        grid10 = [round(x, 2) for x in np.arange(0.80, 1.101, 0.05)]
        cands = {c: book.score({r: spread(eq3[r], c) for r in book.refs}) for c in grid10}
        new, ch = loso(book, cands, "spread", e3)
        rows.append(report("10. spread factor for v3, LOSO (vs v3)", e3, new, ch))
        print(f"   v3 coverage: 50% range {e3.cov50.mean():.3f}, 95% range {e3.cov95.mean():.3f}; "
              f"after idea 10: {new.cov50.mean():.3f}, {new.cov95.mean():.3f}")

    pd.DataFrame(rows).to_csv(ROOT / "backtest/v3_results.csv", index=False)
    return book


def export(out: Path):
    """Official results file for v3 (same format as backtest/run_v2.py): baseline and v1 as in
    results_v2.csv, v2 = the previous 'ensemble', and 'ensemble' = v3 (equal-weight average of the
    regressions, the trees and the quantile autoregression), all on identical forecasts."""
    book = Book()
    lin = {r: mean_members(book.mem[r]["members"]) for r in book.refs}
    v3 = book.score({r: quantile_average([lin[r], book.gbq[r], book.qar[r]]) for r in book.refs})
    v2 = book.score({r: blend(lin[r], book.gbq[r], 0.33) for r in book.refs})  # v2 with today's code
    old = pd.read_csv(ROOT / "backtest/results_v2.csv", dtype={"location": str})
    old = old[old["model"].isin(["baseline", "v1"])]  # unchanged by later fixes (no season clue in them)

    def fmt(d, name):
        return pd.DataFrame({"season": d["season"], "reference_date": d["ref"].dt.date.astype(str),
                             "location": d["location"], "horizon": d["k"] - 1, "observed": d["y"],
                             "wis": d["wis"], "cov50": d["cov50"], "cov95": d["cov95"],
                             "median": d["median"], "model": name})
    new = pd.concat([fmt(v2, "v2"), fmt(v3, "ensemble")])
    key = ["season", "reference_date", "location", "horizon"]
    assert len(old[old.model == "baseline"].merge(new[new.model == "v2"][key], on=key)) == len(v3)
    res = pd.concat([old, new[old.columns]])
    res.to_csv(out, index=False)
    for s_, g in res.groupby("season"):
        m = g.groupby("model")["wis"].mean()
        print(s_, {k: round(m[k] / m["baseline"], 3) for k in ("v1", "v2", "ensemble")})
    m = res.groupby("model")["wis"].mean()
    print("ALL", {k: round(m[k] / m["baseline"], 3) for k in ("v1", "v2", "ensemble")}, "->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--produce-qar", nargs="?", const="qar", default=None,
                    help="cache a family for the 50-state replay: qar (default), qarw, ar6")
    ap.add_argument("--produce-all", nargs="*", default=None, help="replay all 53 places; families mem gbq qar (default all; gbq is slow)")
    ap.add_argument("--export", type=Path, default=None)
    a = ap.parse_args()
    if a.produce_qar:
        produce_qar(a.produce_qar)
    elif a.produce_all is not None:
        produce_all(families=tuple(a.produce_all) or ("mem", "gbq", "qar"))
    elif a.export:
        export(a.export)
    else:
        main()
