"""Round 17 (2026-10-21): spread by place size. Live diagnosis: in states under ~100/week our median
growth matched reality on average, but single-state outcomes swing with count noise, and our 95% range
is ~30% wider (log scale) than the ensemble's. Test: scale log-spread around the median by s_small for
places whose as-published newest week is < T, s_big otherwise; chosen leave-one-season-out.
"""
import numpy as np, pandas as pd, itertools
from backtest.round10 import load, lg, mono
from backtest.eval_all53 import wis_rows, HUB
from backtest.exp_v3 import SEASONS
from backtest.run_backtest import Vintages
keys, Q0, truth = load(); y = keys.y.values; seas = keys.season.values; L = lg(Q0)
V = Vintages(HUB); pub = {}
for r in keys.ref.unique():
    last = r - pd.Timedelta(weeks=1); v = V.as_published(last)
    if v is not None:
        v = v[v.date == last]; pub.update({(str(r.date()), l): x for l, x in zip(v.location, v.value)})
lvl = np.array([pub.get((r, l), np.nan) for r, l in zip(keys.ref_s, keys.location)], float)
base = wis_rows(L, lg(y)); med = L[:, 11:12]
grid = list(itertools.product([.7, .8, .9, 1.0, 1.1], [.8, .9, 1.0, 1.1], [50, 100, 200]))
def sc(p):
    ss, sb, T = p; s = np.where(lvl < T, ss, sb)[:, None]
    return wis_rows(med + s * (L - med), lg(y))
cache = {p: sc(p) for p in grid}
out = np.empty(len(y)); ch = []
for s in SEASONS:
    tr = seas != s; best = min(grid, key=lambda p: cache[p][tr].mean()); out[seas == s] = cache[best][seas == s]; ch.append((s, best))
d = pd.DataFrame({"ref": keys.ref_s, "n": out, "b": base}).groupby("ref").sum(); rng = np.random.default_rng(0)
bs = [(lambda x: x.n.sum() / x.b.sum() - 1)(d.sample(len(d), replace=True, random_state=rng.integers(1e9))) for _ in range(2000)]
lo, hi = np.percentile(bs, [2.5, 97.5]); per = {s: round(out[seas == s].sum() / base[seas == s].sum() - 1, 4) for s in SEASONS}
sm = lvl < 100
print(f"S: {out.sum()/base.sum()-1:+.2%} CI [{lo:+.2%},{hi:+.2%}] small-places {out[sm].sum()/base[sm].sum()-1:+.2%} per-season {per}")
print("chosen", ch); print("KEEP" if hi < 0 and max(per.values()) <= 0 else "fail")
