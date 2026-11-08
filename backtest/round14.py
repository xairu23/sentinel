"""Round 14 (2026-10-14): spatial pooling of growth (UGA INFLAenza's edge: calm weeks, small states).
Post-processing of v3.1, pre-registered, same rule; constants leave-one-season-out:
  S1  shift each state's forecast (log scale) so its median growth moves a share a toward the
      population-weighted median growth of the other states in its HHS region
  S2  as S1 with a larger pull for small states: a_i = a * median_pop / (median_pop + pop_i) * 2
  S3  S1 + pull of the forecast spread toward the region's spread (same a)
Last observed value = as published that week (hub archive).
    python -m backtest.round14
"""
from __future__ import annotations
import numpy as np, pandas as pd
from backtest.round10 import load, lg, mono
from backtest.eval_all53 import HUB, wis_rows
from backtest.exp_v3 import SEASONS
from backtest.leaderboard import load_model
from backtest.run_backtest import Vintages

HHS = {1: "09 23 25 33 44 50", 2: "34 36 72", 3: "10 11 24 42 51 54", 4: "01 12 13 21 28 37 45 47",
       5: "17 18 26 27 39 55", 6: "05 22 35 40 48", 7: "19 20 29 31", 8: "08 30 38 46 49 56",
       9: "04 06 15 32", 10: "02 16 41 53"}
REG = {f: r for r, s in HHS.items() for f in s.split()}


def main():
    keys, Q0, truth = load()
    idx = pd.MultiIndex.from_frame(keys[["ref_s", "location", "horizon"]])
    w = load_model(HUB / "model-output/FluSight-baseline", keys.rename(columns={"ref_s": "ref"})[["ref", "location", "horizon"]])
    w.index = w.index.set_names(["ref_s", "location", "horizon"])
    base = w.reindex(idx).values.astype(float); ok = ~np.isnan(base).any(1)
    keys, Q0, base = keys[ok].reset_index(drop=True), Q0[ok], base[ok]
    y = keys.y.values; h = keys.horizon.values
    pop = pd.read_csv(HUB / "auxiliary-data/locations.csv", dtype={"location": str}).set_index("location").population
    V = Vintages(HUB); last = {}
    for r in keys.ref.unique():
        v = V.as_published(r - pd.Timedelta(weeks=1))
        if v is not None:
            v = v[v.date == r - pd.Timedelta(weeks=1)].set_index("location").value
            for l, val in v.items(): last[(r, l)] = val
    L0 = np.array([last.get((r, l), np.nan) for r, l in zip(keys.ref, keys.location)])
    keys["reg"] = keys.location.map(REG); keys["pop"] = keys.location.map(pop).values
    LQ = lg(Q0); g = LQ[:, 11] - np.log1p(L0); spread = LQ[:, 19] - LQ[:, 3]
    keys["g"] = g; keys["sp"] = spread
    # region average of the OTHER states (population-weighted), per ref & horizon
    def other_mean(col):
        out = np.full(len(keys), np.nan)
        for (r, hh, rg), d in keys[keys.reg.notna() & np.isfinite(keys[col])].groupby(["ref", "horizon", "reg"]):
            tot = (d[col] * d["pop"]).sum(); pw = d["pop"].sum()
            for i in d.index:
                if pw - d.at[i, "pop"] > 0:
                    out[i] = (tot - d.at[i, col] * d.at[i, "pop"]) / (pw - d.at[i, "pop"])
        return out
    gR = other_mean("g"); sR = other_mean("sp")
    has = np.isfinite(gR) & np.isfinite(g)
    medpop = np.median(pop.values)
    def S(a, size=False, spr=False):
        ai = np.full(len(y), a) if not size else np.clip(a * 2 * medpop / (medpop + keys["pop"].values), 0, 1)
        ai = np.where(has, ai, 0)
        shift = ai * (np.nan_to_num(gR) - np.nan_to_num(g))
        L = LQ + shift[:, None]
        if spr:
            med = L[:, [11]]; ratio = np.where(has & (spread > 0), (1 - ai) + ai * np.nan_to_num(sR) / np.where(spread > 0, spread, 1), 1)
            L = med + np.clip(ratio, 0.5, 2)[:, None] * (L - med)
        return np.round(mono(np.expm1(L)))
    sc = lambda Q, m: wis_rows(lg(Q[m]), lg(y[m])).sum()
    M = {"v3.1": np.round(Q0)}
    grid = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    for name, kw in [("S1", {}), ("S2", {"size": True}), ("S3", {"spr": True})]:
        cands = {a: S(a, **kw) for a in grid}; out = np.zeros_like(Q0)
        for s in SEASONS:
            trm = (keys.season != s).values
            a = min(grid, key=lambda a_: sc(cands[a_], trm)); print(f"  {name}: a fitted without {s} = {a}")
            out[~trm] = cands[a][~trm]
        M[name] = out
    W = {m: wis_rows(lg(Q), lg(y)) for m, Q in M.items()}; Wb = wis_rows(lg(base), lg(y)); Cn = {m: wis_rows(Q, y) for m, Q in M.items()}
    wk = keys.ref_s.values; weeks = np.unique(wk)
    A0 = pd.Series(W["v3.1"]).groupby(wk).sum().reindex(weeks).values
    ii = np.random.default_rng(0).integers(0, len(weeks), size=(2000, len(weeks)))
    st = ~keys.location.isin(["US", "11", "72"]).values
    print(f"rows with a region average: {has.mean():.0%}")
    print("| model | vs baseline | change vs v3.1 (95% CI) | " + " | ".join(SEASONS) + " | counts | small states | rule |")
    small = (keys["pop"] < medpop).values
    for m, Q in M.items():
        B = pd.Series(W[m]).groupby(wk).sum().reindex(weeks).values
        lo, hi = np.percentile((B[ii].sum(1) / A0[ii].sum(1) - 1) * 100, [2.5, 97.5])
        ps = {s: (W[m][(keys.season == s).values].sum() / W["v3.1"][(keys.season == s).values].sum() - 1) * 100 for s in SEASONS}
        cnt = (Cn[m][st].sum() / Cn["v3.1"][st].sum() - 1) * 100
        sm = (W[m][small].sum() / W["v3.1"][small].sum() - 1) * 100
        keep = m != "v3.1" and hi < 0 and all(v <= 0 for v in ps.values()) and cnt <= 1
        print(f"| {m} | {W[m].sum()/Wb.sum():.3f} | {(B.sum()/A0.sum()-1)*100:+.1f}% ({lo:+.1f} to {hi:+.1f}) | " + " | ".join(f"{ps[s]:+.1f}%" for s in SEASONS) + f" | {cnt:+.1f}% | {sm:+.1f}% | {'KEEP' if keep else ('-' if m=='v3.1' else 'fails')} |")


if __name__ == "__main__":
    main()
