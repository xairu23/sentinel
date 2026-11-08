"""Round 11 (2026-10-14): v3.1's 95% intervals cover 97% (too wide) yet narrowing everywhere hurt surges
(round 8, idea C). Pre-registered: narrow only where the recent curve is calm.
  N  spread factor f applied when |log change of the last two reported weeks| < c, else 1
     (f, c) chosen leave-one-season-out; uses only data known at forecast time (published values).
  NH same, f separate per horizon (c fixed at the N choice)
    python -m backtest.round11
"""
from __future__ import annotations
import numpy as np, pandas as pd
from backtest.round10 import load, lg, mono
from backtest.eval_all53 import HUB, wis_rows
from backtest.exp_v3 import SEASONS
from backtest.leaderboard import load_model


def main():
    keys, Q0, truth = load()
    idx = pd.MultiIndex.from_frame(keys[["ref_s", "location", "horizon"]])
    w = load_model(HUB / "model-output/FluSight-baseline", keys.rename(columns={"ref_s": "ref"})[["ref", "location", "horizon"]])
    w.index = w.index.set_names(["ref_s", "location", "horizon"])
    base = w.reindex(idx).values.astype(float); ok = ~np.isnan(base).any(1)
    keys, Q0, base = keys[ok].reset_index(drop=True), Q0[ok], base[ok]
    y = keys.y.values; h = keys.horizon.values
    # recent change from the forecast's own published history: use the replay's truth table but only
    # weeks <= ref-1 (final values; approximation noted in the protocol)
    tr = truth.set_index(["location", "date"])["value"]
    def val(l, d): return tr.get((l, d), np.nan)
    ch = np.array([np.log1p(val(l, r - pd.Timedelta(weeks=1))) - np.log1p(val(l, r - pd.Timedelta(weeks=3)))
                   for l, r in zip(keys.location, keys.ref)])
    ch = np.abs(np.nan_to_num(ch, nan=1.0))
    L = lg(Q0); med = L[:, [11]]
    def apply(fs, c):
        f = np.where(ch < c, fs[h] if np.ndim(fs) else fs, 1.0)[:, None]
        return np.round(mono(np.expm1(med + f * (L - med))))
    sc = lambda Q, m: wis_rows(lg(Q[m]), lg(y[m])).sum()
    M = {"v3.1": np.round(Q0)}
    grid = [(f, c) for f in [0.6, 0.7, 0.8, 0.9, 1.0] for c in [0.05, 0.1, 0.2, 0.3]]
    cand = {g: apply(*g) for g in grid}
    outN = np.zeros_like(Q0); outNH = np.zeros_like(Q0)
    for s in SEASONS:
        trm = (keys.season != s).values
        f, c = min(grid, key=lambda g: sc(cand[g], trm))
        fh = []
        for hh in range(4):
            m = trm & (h == hh)
            fh.append(min([0.6, 0.7, 0.8, 0.9, 1.0], key=lambda ff: sc(apply(ff, c), m)))
        print(f"  without {s}: N f={f} c={c}; NH f by horizon={fh}")
        outN[~trm] = cand[(f, c)][~trm]; outNH[~trm] = apply(np.array(fh), c)[~trm]
    M["N"], M["NH"] = outN, outNH
    W = {m: wis_rows(lg(Q), lg(y)) for m, Q in M.items()}; Wb = wis_rows(lg(base), lg(y)); Cn = {m: wis_rows(Q, y) for m, Q in M.items()}
    wk = keys.ref_s.values; weeks = np.unique(wk)
    A0 = pd.Series(W["v3.1"]).groupby(wk).sum().reindex(weeks).values
    ii = np.random.default_rng(0).integers(0, len(weeks), size=(2000, len(weeks)))
    st = ~keys.location.isin(["US", "11", "72"]).values
    print(f"calm share at c=0.1: {(ch<0.1).mean():.0%}")
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
