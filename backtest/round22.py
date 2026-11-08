"""Round 22 (2026-10-30): smoothing the starting point (see V3_PROTOCOL.md).
    python -m backtest.round22
"""
import numpy as np, pandas as pd, itertools
from backtest.round10 import load, lg
from backtest.eval_all53 import wis_rows, HUB
from backtest.exp_v3 import SEASONS
from backtest.run_backtest import Vintages

keys, Q0, truth = load(); y = keys.y.values; seas = keys.season.values; h = keys.horizon.values
V = Vintages(HUB); KS = [3, 4, 5]
newest = np.full(len(keys), np.nan); D = {k: np.full(len(keys), np.nan) for k in KS}
for r in keys.ref.unique():
    last = r - pd.Timedelta(weeks=1); v = V.as_published(last)
    if v is None: continue
    ix = np.where(keys.ref.values == r)[0]
    g = {l: s.sort_values("date").value.values.astype(float) for l, s in v.groupby("location")}
    for i in ix:
        s = g.get(keys.location.values[i])
        if s is None or len(s) < 5 or np.isnan(s[-5:]).any(): continue
        newest[i] = s[-1]
        for k in KS:
            z = np.log1p(s[-k:]); t = np.arange(k) - (k - 1)
            b = np.polyfit(t, z, 1); D[k][i] = np.polyval(b, 0) - z[-1]
ok = ~np.isnan(newest)
print(f"rows {len(keys)}, with as-published history {ok.mean():.3f}")
L = lg(Q0); ly = lg(y); base = wis_rows(L, ly); err = L[:, 11] - ly

# descriptive: lag-1 correlation of h0 median errors, consecutive weeks, same place
e0 = keys.assign(err=err, n=newest)[h == 0].sort_values(["location", "ref"])
e0["prev"] = e0.groupby(["location", "season"]).err.shift(1)
e0 = e0.dropna(subset=["prev"])
for nm, m in [("newest < 100", e0.n < 100), ("newest >= 100", e0.n >= 100)]:
    x = e0[m]
    print(f"h0 lag-1 error correlation, {nm}: {np.corrcoef(x.err, x.prev)[0, 1]:+.3f} (n={len(x)})",
          {s: round(np.corrcoef(x[x.season == s].err, x[x.season == s].prev)[0, 1], 3) for s in SEASONS})

def evaluate(name, grid, fn):
    cache = {p: fn(p) for p in grid}
    out = np.empty(len(y)); ch = []
    for s in SEASONS:
        tr = seas != s; best = min(grid, key=lambda p: cache[p][tr].mean())
        out[seas == s] = cache[best][seas == s]; ch.append((s, best))
    d = pd.DataFrame({"ref": keys.ref_s, "n": out, "b": base}).groupby("ref").sum(); rng = np.random.default_rng(0)
    bs = [(lambda x: x.n.sum() / x.b.sum() - 1)(d.sample(len(d), replace=True, random_state=rng.integers(1e9))) for _ in range(2000)]
    lo, hi = np.percentile(bs, [2.5, 97.5]); per = {s: round(out[seas == s].sum() / base[seas == s].sum() - 1, 4) for s in SEASONS}
    sm = newest < 100
    print(f"{name}: {out.sum()/base.sum()-1:+.2%} CI [{lo:+.2%},{hi:+.2%}] small {out[sm].sum()/base[sm].sum()-1:+.2%} per-season {per}")
    print("  chosen", ch); print("  KEEP" if hi < 0 and max(per.values()) <= 0 else "  fail")
    return out

def shifted(w, k):
    w = np.where(ok, w, 0.0); return wis_rows(L + (w * np.nan_to_num(D[k]))[:, None], ly)
g1 = list(itertools.product([0, .25, .5, .75, 1], [50, 100, 200, 1e9], KS))
evaluate("A1", g1, lambda p: shifted(np.where(newest < p[1], p[0], 0.0), p[2]))
g2 = list(itertools.product([5, 10, 20, 40, 80], KS))
evaluate("A2", g2, lambda p: shifted(p[0] / (p[0] + np.nan_to_num(newest, nan=1e9)), p[1]))
