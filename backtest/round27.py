"""Round 27: ER anchor beyond 200/week (vs v3.3).  python -m backtest.round27"""
import io, contextlib, runpy, itertools, numpy as np, pandas as pd
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_module("backtest.round26c", run_name="x")
keys, seas, L32, ly, D, newest, er, ok = (g[k] for k in ["keys", "seas", "L32", "ly", "D", "newest", "er", "ok"])
from backtest.eval_all53 import wis_rows
from backtest.exp_v3 import SEASONS
def run(a, T):
    w = np.where(ok & (np.nan_to_num(newest, nan=1e12) < T) & (np.nan_to_num(er) >= .01), a, 0.0)
    return wis_rows(L32 + (w * np.nan_to_num(D))[:, None], ly)
base = run(.5, 200)  # v3.3
grid = list(itertools.product([.25, .5, .75], [200, 400, 800, 1e11]))
cache = {p: run(*p) for p in grid}; out = np.empty(len(ly)); ch = []
for s in SEASONS:
    tr = seas != s; best = min(grid, key=lambda p: cache[p][tr].mean()); out[seas == s] = cache[best][seas == s]; ch.append((s, best))
d = pd.DataFrame({"ref": keys.ref_s, "n": out, "b": base}).groupby("ref").sum(); rng = np.random.default_rng(0)
bs = [(lambda x: x.n.sum() / x.b.sum() - 1)(d.sample(len(d), replace=True, random_state=rng.integers(1e9))) for _ in range(2000)]
lo, hi = np.percentile(bs, [2.5, 97.5]); pr = {s: round(float(out[seas == s].sum() / base[seas == s].sum() - 1), 4) for s in SEASONS}
print(f"E4: {out.sum()/base.sum()-1:+.2%} CI [{lo:+.2%},{hi:+.2%}] per-season {pr}"); print("  chosen", ch)
print("  KEEP" if hi < 0 and max(pr.values()) <= 0 else "  fail")
for p in grid: print("  ", p, f"{cache[p].sum()/base.sum()-1:+.2%}")
