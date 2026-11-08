"""Round 10 (2026-10-14): techniques of the best FluSight teams, applied to v3.1 (all 53 places).
Pre-registered; same rule as rounds 3-9. Tuning constants chosen leave-one-season-out.
  M  online quantile calibration per place and horizon (CMU-TimeSeries_Cal / MultiQT): an additive
     offset on sqrt(rate per 100k), updated by pinball-loss gradient steps as truths arrive in season
  K  climatology blend growing with horizon and towards the tails (CMU climate_linear): climatology =
     quantiles of the place's own past seasons, same week +-3, rescaled to the place
  X  wider intervals for targets in the Christmas / New Year weeks (flusion, FluSight 2024 & 2025)
    python -m backtest.round10
"""
from __future__ import annotations
import numpy as np, pandas as pd
from pathlib import Path
from backtest.eval_all53 import HUB, load_family, wis_rows
from backtest.exp_v3 import SEASONS, quantile_average
from backtest.leaderboard import load_model
from backtest.member_cache import mean_members
from backtest.round3 import no_pr
from backtest.run_backtest import final_hosp
from backtest.wis import QUANTILES as QS
lg = lambda a: np.log1p(np.maximum(a, 0))
mono = lambda Q: np.maximum.accumulate(np.maximum(Q, 0), axis=1)


def load():
    F = {f: load_family(f) for f in ["mem", "gbqbf", "qar"]}
    refs = sorted(set.intersection(*[set(v) for v in F.values()]))
    truth = final_hosp()
    tr = truth.set_index(["location", "date"])["value"]
    rows, Q = [], []
    for r in refs:
        v = quantile_average([mean_members(F["mem"][r]["mem"]), F["gbqbf"][r]["gbqbf"], no_pr(F["qar"][r]["qar"])])
        last = r - pd.Timedelta(weeks=1)
        for (loc, k), q in v.items():
            te = last + pd.Timedelta(weeks=k); y = tr.get((loc, te))
            if y is None or np.isnan(y): continue
            rows.append((F["mem"][r]["season"], r, str(r.date()), loc, k - 1, te, float(y))); Q.append(q)
    keys = pd.DataFrame(rows, columns=["season", "ref", "ref_s", "location", "horizon", "target", "y"])
    return keys, np.array(Q), truth


def climatology(keys, truth, Qv):
    """Per row: 23 quantiles of the place's values in earlier seasons at the same week +-3, scaled so the
    climatological median matches the place's current level ratio (keeps it on the place's scale)."""
    t = truth.copy(); t["wk"] = t.date.dt.isocalendar().week.astype(int)
    out = np.full((len(keys), 23), np.nan)
    g = {loc: d for loc, d in t.groupby("location")}
    for i, (loc, ref, te) in enumerate(zip(keys.location, keys.ref, keys.target)):
        d = g[loc]; w = te.isocalendar()[1]
        cutoff = ref - pd.Timedelta(weeks=26)          # strictly earlier seasons
        dd = d[(d.date < cutoff) & ((d.wk - w).abs().clip(upper=53) .where(lambda s: s <= 26, 53 - (d.wk - w).abs()) <= 3)]
        if len(dd) >= 5:
            out[i] = np.quantile(dd.value.values, QS)
    ok = ~np.isnan(out).any(1)
    out[~ok] = Qv[~ok]
    return out, ok


def main():
    keys, Q0, truth = load()
    idx = pd.MultiIndex.from_frame(keys[["ref_s", "location", "horizon"]])
    w = load_model(HUB / "model-output/FluSight-baseline", keys.rename(columns={"ref_s": "ref"})[["ref", "location", "horizon"]])
    w.index = w.index.set_names(["ref_s", "location", "horizon"])
    base = w.reindex(idx).values.astype(float); ok = ~np.isnan(base).any(1)
    keys, Q0, base = keys[ok].reset_index(drop=True), Q0[ok], base[ok]
    y = keys.y.values; h = keys.horizon.values
    pop = pd.read_csv(HUB / "auxiliary-data/locations.csv", dtype={"location": str}).set_index("location").population
    P = keys.location.map(pop).values.astype(float)
    sc = lambda Q, m=None: wis_rows(lg(Q if m is None else Q[m]), lg(y if m is None else y[m]))
    M = {"v3.1": np.round(Q0)}

    # --- M: online MultiQT on sqrt(per 100k)
    def multiqt(lr):
        Z = np.sqrt(np.maximum(Q0, 0) / P[:, None] * 1e5); zy = np.sqrt(y / P * 1e5)
        out = Z.copy()
        for s in SEASONS:
            ix = np.where(keys.season.values == s)[0]
            refs = sorted(keys.ref.values[ix].astype("datetime64[ns]").tolist() and set(keys.ref[ix]))
            off = {}
            for r in refs:
                cur = ix[keys.ref.values[ix] == r]
                for j in cur:
                    out[j] = np.maximum.accumulate(Z[j] + off.get((keys.location[j], h[j]), 0))
                # update with every forecast (any earlier ref) whose target has now been observed by the next ref
                known = ix[(keys.target.values[ix] <= r) & (keys.target.values[ix] > r - pd.Timedelta(weeks=1))]
                for j in known:
                    k_ = (keys.location[j], h[j]); o = off.get(k_, np.zeros(23))
                    grad = (zy[j] <= out[j]).astype(float) - QS   # pinball gradient wrt quantile
                    off[k_] = o - lr * grad
        return np.round(mono((out ** 2) * P[:, None] / 1e5))
    # --- K: climatology blend
    C, cok = climatology(keys, truth, Q0)
    tail = np.abs(QS - 0.5) * 2                          # 0 at median, ~1 at the extremes
    def kblend(a):
        wq = np.clip(a * (0.25 + 0.75 * tail)[None, :] * ((h + 1) / 4)[:, None], 0, 1) * cok[:, None]
        return np.round(mono((1 - wq) * Q0 + wq * C))
    # --- X: Christmas widening
    tw = keys.target.dt.isocalendar().week.astype(int).values
    xmas = np.isin(tw, [52, 53, 1])
    def xwide(f):
        L = lg(Q0); med = L[:, [11]]; s = np.where(xmas, f, 1.0)[:, None]
        return np.round(mono(np.expm1(med + s * (L - med))))
    grids = {"M": (multiqt, [0.0, 0.005, 0.01, 0.02, 0.04]), "K": (kblend, [0.0, 0.1, 0.2, 0.3, 0.5]), "X": (xwide, [1.0, 1.1, 1.2, 1.35, 1.5])}
    for name, (fn, grid) in grids.items():
        cand = {g: fn(g) for g in grid}; out = np.zeros_like(Q0)
        for s in SEASONS:
            tr = (keys.season != s).values
            best = min(grid, key=lambda g: sc(cand[g], tr).sum())
            print(f"  {name}: constant fitted without {s} = {best}")
            out[~tr] = cand[best][~tr]
        M[name] = out
    W = {m: sc(Q) for m, Q in M.items()}; Wb = wis_rows(lg(base), lg(y)); Cn = {m: wis_rows(Q, y) for m, Q in M.items()}
    wk = keys.ref_s.values; weeks = np.unique(wk)
    A0 = pd.Series(W["v3.1"]).groupby(wk).sum().reindex(weeks).values
    ii = np.random.default_rng(0).integers(0, len(weeks), size=(2000, len(weeks)))
    st = ~keys.location.isin(["US", "11", "72"]).values
    print(f"\nclimatology available for {cok.mean():.0%} of rows; Christmas-week targets {xmas.mean():.1%}")
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
