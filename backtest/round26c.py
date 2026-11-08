"""Round 26c: ER anchor only when the ER flu share is high enough.  python -m backtest.round26c"""
import io, contextlib, runpy, itertools, numpy as np, pandas as pd
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_module("backtest.round26b", run_name="x")
keys, D, newest, seas, L32, ly, base, fin, src = (g[k] for k in ["keys", "D", "newest", "seas", "L32", "ly", "base", "fin", "src"])
from backtest.eval_all53 import wis_rows
from backtest.exp_v3 import SEASONS
ed_by_asof, V, corr, pooled = g["ed_by_asof"], g["V"], g["corr"], g["pooled"]
# newest ER share as published (archived where available, else final / typical revision), same as 26b
er = np.full(len(keys), np.nan)
for r in keys.ref.unique():
    last = r - pd.Timedelta(weeks=1); asof = V.t[V.t.date == last].as_of.min(); Eh = ed_by_asof.get(asof)
    for i in np.where(keys.ref.values == r)[0]:
        l = keys.location.values[i]
        er[i] = Eh.get((l, last), np.nan) if (Eh is not None and (l, last) in Eh.index) else fin.get((l, last), np.nan) / np.exp(corr.get(l, pooled))
ok = ~np.isnan(D); small = np.nan_to_num(newest, nan=1e9) < 100
def run(a, T, gate):
    w = np.where(ok & (np.nan_to_num(newest, nan=1e9) < T) & (np.nan_to_num(er) >= gate), a, 0.0)
    return wis_rows(L32 + (w * np.nan_to_num(D))[:, None], ly)
grid = list(itertools.product([0, .25, .5, .75, 1], [50, 100, 200], [0, .005, .01, .02]))
cache = {p: run(*p) for p in grid}; out = np.empty(len(ly)); ch = []
for s in SEASONS:
    tr = seas != s; best = min(grid, key=lambda p: cache[p][tr].mean()); out[seas == s] = cache[best][seas == s]; ch.append((s, best))
d = pd.DataFrame({"ref": keys.ref_s, "n": out, "b": base}).groupby("ref").sum(); rng = np.random.default_rng(0)
bs = [(lambda x: x.n.sum() / x.b.sum() - 1)(d.sample(len(d), replace=True, random_state=rng.integers(1e9))) for _ in range(2000)]
lo, hi = np.percentile(bs, [2.5, 97.5]); pr = {s: round(float(out[seas == s].sum() / base[seas == s].sum() - 1), 4) for s in SEASONS}
mo = keys.target.dt.month.values; early = np.isin(mo, [10, 11])
print(f"E3: {out.sum()/base.sum()-1:+.2%} CI [{lo:+.2%},{hi:+.2%}] per-season {pr} Oct-Nov {out[early].sum()/base[early].sum()-1:+.2%}")
print("  chosen", ch); print("  KEEP" if hi < 0 and max(pr.values()) <= 0 else "  fail")
for p in [(.5, 200, .005), (.5, 200, .01), (.5, 200, .02)]:
    x = cache[p]; print(" fixed", p, f"{x.sum()/base.sum()-1:+.2%}", {s: f"{x[seas == s].sum()/base[seas == s].sum()-1:+.2%}" for s in SEASONS})
