"""Round 9 (2026-10-14): follow-up of the team study. Our three families make highly correlated
errors (0.85-0.96) and QAR is the weakest member with a downward bias that grows with horizon.
Pre-registered, same rule, vs v3.1:
  W  family weights chosen leave-one-season-out on a grid (weights sum to 1)
  H  QAR only for horizons 0-1 (dropped where its bias is largest)
  T2 trees counted twice (regressions, trees, trees, QAR)
    python -m backtest.round9
"""
from __future__ import annotations
import itertools, numpy as np, pandas as pd
from backtest.eval_all53 import HUB, load_family, wis_rows
from backtest.exp_v3 import SEASONS, quantile_average
from backtest.leaderboard import load_model
from backtest.member_cache import mean_members
from backtest.round3 import no_pr
from backtest.run_backtest import final_hosp
lg = lambda a: np.log1p(np.maximum(a, 0))

def main():
    F = {f: load_family(f) for f in ["mem", "gbqbf", "qar"]}
    refs = sorted(set.intersection(*[set(v) for v in F.values()]))
    season = {r: F["mem"][r]["season"] for r in refs}
    truth = final_hosp().set_index(["location", "date"])["value"]
    rows, R, G, Qr = [], [], [], []
    for r in refs:
        lin = mean_members(F["mem"][r]["mem"]); g = F["gbqbf"][r]["gbqbf"]; q = no_pr(F["qar"][r]["qar"])
        last = r - pd.Timedelta(weeks=1)
        for (loc, k) in lin:
            y = truth.get((loc, last + pd.Timedelta(weeks=k)))
            if y is None or np.isnan(y) or (loc, k) not in g: continue
            rows.append((season[r], str(r.date()), loc, k - 1, float(y)))
            R.append(lin[(loc, k)]); G.append(g[(loc, k)]); Qr.append(q.get((loc, k), np.full(23, np.nan)))
    keys = pd.DataFrame(rows, columns=["season", "ref", "location", "horizon", "y"])
    R, G, Qr = map(np.array, (R, G, Qr))
    hasq = ~np.isnan(Qr).any(1)
    Qr = np.where(hasq[:, None], Qr, (R + G) / 2)
    idx = pd.MultiIndex.from_frame(keys[["ref", "location", "horizon"]])
    w = load_model(HUB / "model-output/FluSight-baseline", keys[["ref", "location", "horizon"]]); base = w.reindex(idx).values.astype(float)
    ok = ~np.isnan(base).any(1)
    keys, R, G, Qr, base = keys[ok].reset_index(drop=True), R[ok], G[ok], Qr[ok], base[ok]
    y = keys.y.values; h = keys.horizon.values
    def comb(wr, wg, wq, qmask=None):
        wq_eff = np.where(qmask, wq, 0) if qmask is not None else np.full(len(y), wq)
        wq_eff = np.where(hasq[ok], wq_eff, 0)
        s = wr + wg + wq_eff
        Q = (wr * R + wg * G + wq_eff[:, None] * Qr) / np.where(s > 0, s, 1)[:, None]
        Q = np.where((s > 0)[:, None], Q, (R + G) / 2)
        return np.round(np.maximum.accumulate(np.maximum(Q, 0), axis=1))
    sc = lambda Q: wis_rows(lg(Q), lg(y))
    M = {"v3.1": comb(1, 1, 1), "H": comb(1, 1, 1, h <= 1), "T2": comb(1, 2, 1)}
    grid = [(a, b, round(1 - a - b, 2)) for a in np.arange(0, 1.01, .1) for b in np.arange(0, 1.01, .1) if a + b <= 1.0001]
    Wq = np.zeros_like(M["v3.1"])
    for s in SEASONS:
        tr = (keys.season != s).values
        best = min(grid, key=lambda g_: sc(comb(*g_)[tr]).sum() if False else wis_rows(lg(comb(*g_)[tr]), lg(y[tr])).sum())
        print(f"  weights (reg, trees, qar) fitted without {s}: {best}")
        Wq[~tr] = comb(*best)[~tr]
    M["W"] = Wq
    W = {m: sc(Q) for m, Q in M.items()}; Wb = wis_rows(lg(base), lg(y)); Cn = {m: wis_rows(Q, y) for m, Q in M.items()}
    wk = keys.ref.values; weeks = np.unique(wk)
    A0 = pd.Series(W["v3.1"]).groupby(wk).sum().reindex(weeks).values
    ii = np.random.default_rng(0).integers(0, len(weeks), size=(2000, len(weeks)))
    st = ~keys.location.isin(["US", "11", "72"]).values
    print("\n| model | vs baseline | change vs v3.1 (95% CI) | " + " | ".join(SEASONS) + " | counts | rule |")
    for m in M:
        B = pd.Series(W[m]).groupby(wk).sum().reindex(weeks).values
        lo, hi = np.percentile((B[ii].sum(1) / A0[ii].sum(1) - 1) * 100, [2.5, 97.5])
        ps = {s: (W[m][(keys.season == s).values].sum() / W["v3.1"][(keys.season == s).values].sum() - 1) * 100 for s in SEASONS}
        cnt = (Cn[m][st].sum() / Cn["v3.1"][st].sum() - 1) * 100
        keep = m != "v3.1" and hi < 0 and all(v <= 0 for v in ps.values()) and cnt <= 1
        print(f"| {m} | {W[m].sum()/Wb.sum():.3f} | {(B.sum()/A0.sum()-1)*100:+.1f}% ({lo:+.1f} to {hi:+.1f}) | " + " | ".join(f"{ps[s]:+.1f}%" for s in SEASONS) + f" | {cnt:+.1f}% | {'KEEP' if keep else ('-' if m=='v3.1' else 'fails')} |")

if __name__ == "__main__":
    main()
