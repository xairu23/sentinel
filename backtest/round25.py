"""Round 25 (2026-10-30): tail factors for the trees and the regressions, on top of v3.2.  python -m backtest.round25"""
import numpy as np, pandas as pd, itertools
from backtest.round10 import lg
from backtest.eval_all53 import wis_rows
from backtest.exp_v3 import SEASONS
from backtest.fam_rows import get, combine
from models.engine import qar_tail

keys, Q0, fam, newest = get(); y = keys.y.values; seas = keys.season.values; h = keys.horizon.values
ly = lg(y); small = np.nan_to_num(newest, nan=1e9) < 100
def tf(Q, fl, fu):
    L = lg(Q); m = L[:, 11:12]; d = L - m; return np.where(np.isnan(Q), np.nan, np.expm1(m + np.where(d < 0, fl, fu) * d))
F32 = dict(fam); F32["qar"] = tf(fam["qar"], .4, 1.0)
base = wis_rows(lg(combine(F32)), ly)
for nm, f in [("reg", "reg"), ("tree", "tree"), ("qar v3.2", "qar")]:
    L = lg(F32[f]); ok = ~np.isnan(L).any(1)
    print(f"{nm:9s} below2.5 {np.mean(ly[ok] < L[ok,1]):.3f} above97.5 {np.mean(ly[ok] > L[ok,21]):.3f} cover95 {np.mean((ly[ok]>=L[ok,1])&(ly[ok]<=L[ok,21])):.3f}")
def evaluate(name, grid, fn):
    cache = {p: fn(p) for p in grid}; out = np.empty(len(y)); ch = []
    for s in SEASONS:
        for hh in range(4):
            tr = (seas != s) & (h == hh); te = (seas == s) & (h == hh)
            best = min(grid, key=lambda p: cache[p][tr].mean()); out[te] = cache[best][te]; ch.append((s, hh, best))
    d = pd.DataFrame({"ref": keys.ref_s, "n": out, "b": base}).groupby("ref").sum(); rng = np.random.default_rng(0)
    bs = [(lambda x: x.n.sum() / x.b.sum() - 1)(d.sample(len(d), replace=True, random_state=rng.integers(1e9))) for _ in range(2000)]
    lo, hi = np.percentile(bs, [2.5, 97.5]); per = {s: round(float(out[seas == s].sum() / base[seas == s].sum() - 1), 4) for s in SEASONS}
    print(f"{name}: {out.sum()/base.sum()-1:+.2%} CI [{lo:+.2%},{hi:+.2%}] small {out[small].sum()/base[small].sum()-1:+.2%} per-season {per}")
    print("  chosen", pd.Series([c[2] for c in ch]).value_counts().to_dict()); print("  KEEP" if hi < 0 and max(per.values()) <= 0 else "  fail")
G = list(itertools.product([.6, .8, 1], [1, 1.2, 1.4, 1.6]))
def run(**fs):
    f = dict(F32)
    for k, (a, b) in fs.items(): f[k] = tf(F32[k], a, b)
    return wis_rows(lg(combine(f)), ly)
evaluate("R1 trees", G, lambda p: run(tree=p))
evaluate("R2 regressions", G, lambda p: run(reg=p))
G3 = [(a, b) for a in G for b in G if a[0] == b[0]]  # same lower factor for both, upper factors free (keeps the grid small)
evaluate("R3 both", G3, lambda p: run(tree=p[0], reg=p[1]))
