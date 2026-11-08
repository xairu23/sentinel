"""Round 13 (query values as published that week, from the hub archive)
Round 13 (2026-10-14): round 12 showed the analog member makes different errors (correlation 0.67 with
v3.1) and V4 gained 3.1% overall (CI -5.2 to -0.9) but failed the per-season rule (+0.3% in two seasons)
and counts. Pre-registered refinements, all chosen leave-one-season-out:
  B1  analog library adds ILI+ curves (2010-2020); K from {100, 200, 400, 800}; member weight w from
      {0.15, 0.2, 0.25, 0.33}
  B2  B1 + neighbours restricted to the same time of year (season week within +-6)
  (each compared with v3.1 under the same rule)
    python -m backtest.round13
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from backtest.round10 import load, lg, mono
from backtest.eval_all53 import HUB, wis_rows
from backtest.exp_v3 import SEASONS
from backtest.leaderboard import load_model
from backtest.wis import QUANTILES as QS
from backtest.run_backtest import Vintages
ROOT = Path(__file__).resolve().parents[1]
EPS = 0.05


def sw(week):  # season week: week 40 -> 0
    return (np.asarray(week) - 40) % 52


def library(truth, pop):
    cut, P, F, W = [], [], [], []
    t = truth.copy(); t["rate"] = t.value / t.location.map(pop) * 1e5
    for loc, d in t.dropna(subset=["rate"]).groupby("location"):
        d = d.sort_values("date"); z = np.log(d.rate.values + EPS); dt = d.date.values
        wk = d.date.dt.isocalendar().week.values.astype(int)
        for i in range(4, len(z) - 4):
            cut.append(dt[i + 4]); P.append(np.diff(z[i - 4:i + 1])); F.append(z[i + 1:i + 5] - z[i]); W.append(sw(wk[i]))
    fs = pd.read_csv(ROOT / "data/raw/sentinel_history.csv")
    for (src, loc), d in fs.groupby(["source", "location"]):
        d = d.sort_values("epiweek"); z = np.log(d.v1.values + EPS); ew = d.epiweek.values
        for i in range(4, len(z) - 4):
            if ew[i + 4] - ew[i - 4] > 20: continue
            cut.append(np.datetime64("2020-06-01")); P.append(np.diff(z[i - 4:i + 1])); F.append(z[i + 1:i + 5] - z[i]); W.append(sw(ew[i] % 100))
    return np.array(cut, dtype="datetime64[ns]"), np.vstack(P), np.vstack(F), np.array(W)


def main():
    keys, Q0, truth = load()
    idx = pd.MultiIndex.from_frame(keys[["ref_s", "location", "horizon"]])
    w = load_model(HUB / "model-output/FluSight-baseline", keys.rename(columns={"ref_s": "ref"})[["ref", "location", "horizon"]])
    w.index = w.index.set_names(["ref_s", "location", "horizon"])
    base = w.reindex(idx).values.astype(float); ok = ~np.isnan(base).any(1)
    keys, Q0, base = keys[ok].reset_index(drop=True), Q0[ok], base[ok]
    y = keys.y.values; h = keys.horizon.values
    pop = pd.read_csv(HUB / "auxiliary-data/locations.csv", dtype={"location": str}).set_index("location").population
    cut, PAST, FUT, SW = library(truth, pop)
    print(f"library windows: {len(cut)}")
    VIN = Vintages(HUB); vcache = {}
    Ks = [100, 200, 400, 800]; ws = [0.15, 0.2, 0.25, 0.33]
    AN = {(v, K): np.full_like(Q0, np.nan) for v in ["B1", "B2"] for K in Ks}
    for (loc, ref), g in keys.groupby(["location", "ref"]):
        last = ref - pd.Timedelta(weeks=1)
        if last not in vcache:
            v_ = VIN.as_published(last)
            vcache[last] = None if v_ is None else v_.set_index(["location", "date"])["value"]
        tr = vcache[last]
        if tr is None: continue
        vals = [tr.get((loc, last - pd.Timedelta(weeks=j)), np.nan) for j in range(4, -1, -1)]
        if np.isnan(vals).any(): continue
        z = np.log(np.array(vals) / pop[loc] * 1e5 + EPS); q = np.diff(z)
        base_m = cut <= np.datetime64(ref - pd.Timedelta(weeks=26))
        s_now = sw(int(last.isocalendar()[1]))
        dsw = np.minimum(np.abs(SW - s_now), 52 - np.abs(SW - s_now))
        for v, m in [("B1", base_m), ("B2", base_m & (dsw <= 6))]:
            d = np.sqrt(((PAST[m] - q) ** 2).sum(1)); Fm = FUT[m]; order = np.argsort(d)
            for K in Ks:
                nn = order[:K]; wt = 1 / (d[nn] + 0.05); wt /= wt.sum()
                for j, hh in zip(g.index, g.horizon):
                    fut = Fm[nn, hh]; o = np.argsort(fut); cw = np.cumsum(wt[o])
                    qz = np.interp(QS, cw - wt[o] / 2, fut[o])
                    AN[(v, K)][j] = np.maximum(np.exp(z[-1] + qz) - EPS, 0) * pop[loc] / 1e5
    for k in AN:
        bad = np.isnan(AN[k]).any(1); AN[k][bad] = Q0[bad]
    sc = lambda Q, mm: wis_rows(lg(Q[mm]), lg(y[mm])).sum()
    M = {"v3.1": np.round(Q0)}
    for v in ["B1", "B2"]:
        cands = {(K, wt): np.round(mono((1 - wt) * Q0 + wt * AN[(v, K)])) for K in Ks for wt in ws}
        out = np.zeros_like(Q0)
        for s in SEASONS:
            trm = (keys.season != s).values
            best = min(cands, key=lambda c: sc(cands[c], trm))
            print(f"  {v}: (K, weight) fitted without {s} = {best}")
            out[~trm] = cands[best][~trm]
        M[v] = out
    W = {m: wis_rows(lg(Q), lg(y)) for m, Q in M.items()}; Wb = wis_rows(lg(base), lg(y)); Cn = {m: wis_rows(Q, y) for m, Q in M.items()}
    wk = keys.ref_s.values; weeks = np.unique(wk)
    A0 = pd.Series(W["v3.1"]).groupby(wk).sum().reindex(weeks).values
    ii = np.random.default_rng(0).integers(0, len(weeks), size=(2000, len(weeks)))
    st = ~keys.location.isin(["US", "11", "72"]).values
    print("| model | vs baseline | change vs v3.1 (95% CI) | " + " | ".join(SEASONS) + " | counts | 95% cover | rule |")
    for m, Q in M.items():
        B = pd.Series(W[m]).groupby(wk).sum().reindex(weeks).values
        lo, hi = np.percentile((B[ii].sum(1) / A0[ii].sum(1) - 1) * 100, [2.5, 97.5])
        ps = {s: (W[m][(keys.season == s).values].sum() / W["v3.1"][(keys.season == s).values].sum() - 1) * 100 for s in SEASONS}
        cnt = (Cn[m][st].sum() / Cn["v3.1"][st].sum() - 1) * 100
        cov = ((Q[:, 1] <= y) & (y <= Q[:, 21])).mean()
        keep = m != "v3.1" and hi < 0 and all(v <= 0 for v in ps.values()) and cnt <= 1
        print(f"| {m} | {W[m].sum()/Wb.sum():.3f} | {(B.sum()/A0.sum()-1)*100:+.1f}% ({lo:+.1f} to {hi:+.1f}) | " + " | ".join(f"{ps[s]:+.1f}%" for s in SEASONS) + f" | {cnt:+.1f}% | {cov:.2f} | {'KEEP' if keep else ('-' if m=='v3.1' else 'fails')} |")


if __name__ == "__main__":
    main()
