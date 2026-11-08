"""Round 12 (2026-10-14): an analog member (NAU-Copycat / Scenariocast idea), structurally different from
our three statistical families. Pre-registered, same rule, vs v3.1:
  AN  analog forecast alone: match the last 4 weekly log-growth steps of a place (rate per 100k) against
      every 4-week window of past curves (NHSN of earlier seasons + FluSurv-NET 2010-2020), take the
      K nearest, and use their next 4 weeks of growth (quantiles over neighbours, distance-weighted)
  V4  v3.1 + AN as a 4th equal member
  V4h AN as a 4th member only for horizons 2-3 (where turning points matter most)
  K (neighbours) chosen leave-one-season-out from {25, 50, 100, 200}
    python -m backtest.round12
"""
from __future__ import annotations
import numpy as np, pandas as pd
from backtest.round10 import load, lg, mono
from backtest.eval_all53 import HUB, wis_rows
from backtest.exp_v3 import SEASONS
from backtest.leaderboard import load_model
from backtest.wis import QUANTILES as QS
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
EPS = 0.05


def library(truth, pop):
    rows = []  # (cutoff_date, past4[4], fut4[4])
    t = truth.copy(); t["rate"] = t.value / t.location.map(pop) * 1e5
    for loc, d in t.dropna(subset=["rate"]).groupby("location"):
        d = d.sort_values("date"); z = np.log(d.rate.values + EPS); dt = d.date.values
        for i in range(4, len(z) - 4):
            rows.append((dt[i + 4], np.diff(z[i - 4:i + 1]), z[i + 1:i + 5] - z[i]))
    fs = pd.read_csv(ROOT / "data/raw/sentinel_history.csv")
    fs = fs[fs.source == "flusurv"]
    for loc, d in fs.groupby("location"):
        d = d.sort_values("epiweek")
        z = np.log(d.v1.values + EPS); ew = d.epiweek.values
        for i in range(4, len(z) - 4):
            if ew[i + 4] - ew[i - 4] > 20: continue  # gap (off-season)
            rows.append((np.datetime64("2020-06-01"), np.diff(z[i - 4:i + 1]), z[i + 1:i + 5] - z[i]))
    cut = np.array([r[0] for r in rows], dtype="datetime64[ns]")
    return cut, np.vstack([r[1] for r in rows]), np.vstack([r[2] for r in rows])


def main():
    keys, Q0, truth = load()
    idx = pd.MultiIndex.from_frame(keys[["ref_s", "location", "horizon"]])
    w = load_model(HUB / "model-output/FluSight-baseline", keys.rename(columns={"ref_s": "ref"})[["ref", "location", "horizon"]])
    w.index = w.index.set_names(["ref_s", "location", "horizon"])
    base = w.reindex(idx).values.astype(float); ok = ~np.isnan(base).any(1)
    keys, Q0, base = keys[ok].reset_index(drop=True), Q0[ok], base[ok]
    y = keys.y.values; h = keys.horizon.values
    pop = pd.read_csv(HUB / "auxiliary-data/locations.csv", dtype={"location": str}).set_index("location").population
    cut, PAST, FUT = library(truth, pop)
    tr = truth.set_index(["location", "date"])["value"]
    Ks = [25, 50, 100, 200]
    AN = {K: np.full_like(Q0, np.nan) for K in Ks}
    for (loc, ref), g in keys.groupby(["location", "ref"]):
        last = ref - pd.Timedelta(weeks=1)
        vals = [tr.get((loc, last - pd.Timedelta(weeks=j)), np.nan) for j in range(4, -1, -1)]
        if np.isnan(vals).any(): continue
        z = np.log(np.array(vals) / pop[loc] * 1e5 + EPS); q = np.diff(z)
        m = cut <= np.datetime64(ref - pd.Timedelta(weeks=26))
        d = np.sqrt(((PAST[m] - q) ** 2).sum(1)); F = FUT[m]
        order = np.argsort(d)
        for K in Ks:
            nn = order[:K]; wt = 1 / (d[nn] + 0.05); wt /= wt.sum()
            for j, hh in zip(g.index, g.horizon):
                fut = F[nn, hh]                      # growth to target (hh+1 weeks ahead of last obs)
                o = np.argsort(fut); cw = np.cumsum(wt[o])
                qz = np.interp(QS, cw - wt[o] / 2, fut[o])
                AN[K][j] = np.maximum(np.exp(z[-1] + qz) - EPS, 0) * pop[loc] / 1e5
    sc = lambda Q, mm: wis_rows(lg(Q[mm]), lg(y[mm])).sum()
    has = ~np.isnan(AN[50]).any(1)
    for K in Ks: AN[K][~has] = Q0[~has]
    cands = {}
    for K in Ks:
        cands[("AN", K)] = np.round(mono(AN[K]))
        cands[("V4", K)] = np.round(mono((3 * Q0 + AN[K]) / 4))
        hm = (h >= 2)[:, None]
        cands[("V4h", K)] = np.round(mono(np.where(hm, (3 * Q0 + AN[K]) / 4, Q0)))
    M = {"v3.1": np.round(Q0)}
    for name in ["AN", "V4", "V4h"]:
        out = np.zeros_like(Q0)
        for s in SEASONS:
            trm = (keys.season != s).values
            K = min(Ks, key=lambda k: sc(cands[(name, k)], trm))
            print(f"  {name}: K fitted without {s} = {K}")
            out[~trm] = cands[(name, K)][~trm]
        M[name] = out
    W = {m: wis_rows(lg(Q), lg(y)) for m, Q in M.items()}; Wb = wis_rows(lg(base), lg(y)); Cn = {m: wis_rows(Q, y) for m, Q in M.items()}
    wk = keys.ref_s.values; weeks = np.unique(wk)
    A0 = pd.Series(W["v3.1"]).groupby(wk).sum().reindex(weeks).values
    ii = np.random.default_rng(0).integers(0, len(weeks), size=(2000, len(weeks)))
    st = ~keys.location.isin(["US", "11", "72"]).values
    E = pd.DataFrame({"v": np.log1p(Q0[:, 11]) - np.log1p(y), "a": np.log1p(np.nan_to_num(AN[50][:, 11])) - np.log1p(y)})
    print(f"analog available {has.mean():.0%}; error correlation analog vs v3.1 = {E.corr().iloc[0,1]:.2f}")
    print("| model | vs baseline | change vs v3.1 (95% CI) | " + " | ".join(SEASONS) + " | counts | rule |")
    for m, Q in M.items():
        B = pd.Series(W[m]).groupby(wk).sum().reindex(weeks).values
        lo, hi = np.percentile((B[ii].sum(1) / A0[ii].sum(1) - 1) * 100, [2.5, 97.5])
        ps = {s: (W[m][(keys.season == s).values].sum() / W["v3.1"][(keys.season == s).values].sum() - 1) * 100 for s in SEASONS}
        cnt = (Cn[m][st].sum() / Cn["v3.1"][st].sum() - 1) * 100
        keep = m != "v3.1" and hi < 0 and all(v <= 0 for v in ps.values()) and cnt <= 1
        print(f"| {m} | {W[m].sum()/Wb.sum():.3f} | {(B.sum()/A0.sum()-1)*100:+.1f}% ({lo:+.1f} to {hi:+.1f}) | " + " | ".join(f"{ps[s]:+.1f}%" for s in SEASONS) + f" | {cnt:+.1f}% | {'KEEP' if keep else ('-' if m=='v3.1' else 'fails')} |")


if __name__ == "__main__":
    main()
