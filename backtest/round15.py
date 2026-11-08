"""Round 15 (2026-10-21): fix the live overshoot seen in weeks 1-2 (we sit above truth in 56-71% of
states, mostly small Midwest states during the early rise; ensemble bias ~0).
Pre-registered; same rule as rounds 3-14 (95% bootstrap CI over refs below 0, no season worse,
constants chosen leave-one-season-out).
  D  damp positive growth for small places: shift quantiles down by a*growth when the place's own
     nowcast level is under T admissions/week
  B  constant log shift -c on rows whose predicted growth is positive (all places)
  E  D restricted to early season (Oct-Nov refs)
"""
import numpy as np, pandas as pd, itertools
from backtest.round10 import load, lg, mono
from backtest.eval_all53 import wis_rows
from backtest.exp_v3 import SEASONS

keys, Q0, truth = load()
y = keys.y.values; L = lg(Q0); mid = 11
# anchor = newest week as published on the forecast Wednesday (what we actually saw)
from backtest.run_backtest import Vintages
from backtest.eval_all53 import HUB
V = Vintages(HUB); pub = {}
for r in keys.ref.unique():
    last = r - pd.Timedelta(weeks=1); v = V.as_published(last)
    if v is not None:
        v = v[v.date == last]; pub.update({(str(r.date()), l): x for l, x in zip(v.location, v.value)})
a_ = lg(np.array([pub.get((r, l), np.nan) for r, l in zip(keys.ref_s, keys.location)], float))
print("anchor coverage", np.isfinite(a_).mean())
g = np.nan_to_num(L[:, mid] - a_)                       # predicted log growth vs nowcast
lvl = np.expm1(np.nan_to_num(a_))
early = keys.ref.dt.month.isin([10, 11]).values
base = wis_rows(L, lg(y))

def apply(kind, p):
    if kind == "D": a, T = p; s = a * np.clip(g, 0, None) * (lvl < T)
    if kind == "E": a, T = p; s = a * np.clip(g, 0, None) * (lvl < T) * early
    if kind == "B": c, = p; s = c * (g > 0)
    return wis_rows(lg(mono(np.expm1(L - s[:, None]))), lg(y))

grids = {"D": list(itertools.product([.1, .2, .3, .5], [50, 100, 200, 1e9])),
         "E": list(itertools.product([.1, .2, .3, .5], [50, 100, 200, 1e9])),
         "B": [(c,) for c in [.02, .04, .06, .1]]}
seas = keys.season.values
for kind, grid in grids.items():
    cache = {p: apply(kind, p) for p in grid}
    out = np.empty(len(y)); chosen = []
    for s in SEASONS:
        tr = seas != s
        best = min(grid, key=lambda p: cache[p][tr].mean())
        out[seas == s] = cache[best][seas == s]; chosen.append((s, best))
    # bootstrap over refs of relative change
    d = pd.DataFrame({"ref": keys.ref_s, "n": out, "b": base}).groupby("ref").sum()
    rng = np.random.default_rng(0); bs = []
    for _ in range(2000):
        x = d.sample(len(d), replace=True, random_state=rng.integers(1e9)); bs.append(x.n.sum() / x.b.sum() - 1)
    lo, hi = np.percentile(bs, [2.5, 97.5]); rel = out.sum() / base.sum() - 1
    per = {s: round(out[seas == s].sum() / base[seas == s].sum() - 1, 4) for s in SEASONS}
    eh = out[early].sum() / base[early].sum() - 1
    ok = hi < 0 and max(per.values()) <= 0
    print(f"{kind}: {rel:+.2%} CI [{lo:+.2%},{hi:+.2%}] early-season {eh:+.2%} per-season {per} chosen {chosen} -> {'KEEP' if ok else 'fail'}")

# O: online season-so-far bias correction. For ref r, use errors of earlier forecasts (same season)
# whose target week is already observed by r's data (target <= r - 1 week); shift = a * their median.
E0 = L[:, mid] - lg(y)
shift_cache = np.zeros(len(y))
for s in SEASONS:
    ix = np.where(seas == s)[0]
    for r in sorted(set(keys.ref[ix])):
        cur = ix[keys.ref.values[ix] == np.datetime64(r)]
        past = ix[(keys.target.values[ix] <= np.datetime64(r - pd.Timedelta(weeks=1)))]
        if len(past) >= 50: shift_cache[cur] = np.median(E0[past])
grid = [.25, .5, .75, 1.0]
cache = {a: wis_rows(lg(mono(np.expm1(L - a * shift_cache[:, None]))), lg(y)) for a in grid}
out = np.empty(len(y)); ch = []
for s in SEASONS:
    tr = seas != s; best = min(grid, key=lambda a: cache[a][tr].mean())
    out[seas == s] = cache[best][seas == s]; ch.append((s, best))
d = pd.DataFrame({"ref": keys.ref_s, "n": out, "b": base}).groupby("ref").sum()
rng = np.random.default_rng(0)
bs = [(lambda x: x.n.sum() / x.b.sum() - 1)(d.sample(len(d), replace=True, random_state=rng.integers(1e9))) for _ in range(2000)]
lo, hi = np.percentile(bs, [2.5, 97.5]); per = {s: round(out[seas == s].sum() / base[seas == s].sum() - 1, 4) for s in SEASONS}
print(f"O: {out.sum()/base.sum()-1:+.2%} CI [{lo:+.2%},{hi:+.2%}] early {out[early].sum()/base[early].sum()-1:+.2%} per-season {per} chosen {ch} -> {'KEEP' if hi<0 and max(per.values())<=0 else 'fail'}")
