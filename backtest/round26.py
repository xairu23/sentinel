"""Round 26 (2026-10-30): ER-visit anchor for small states, on top of v3.2.  python -m backtest.round26"""
import numpy as np, pandas as pd, itertools
from backtest.round10 import lg
from backtest.eval_all53 import wis_rows, HUB
from backtest.exp_v3 import SEASONS
from backtest.fam_rows import get, combine
from backtest.run_backtest import Vintages
from pathlib import Path

keys, Q0, fam, newest = get(); y = keys.y.values; seas = keys.season.values; h = keys.horizon.values; ly = lg(y)
def tf(Q, fl, fu):
    L = lg(Q); m = L[:, 11:12]; d = L - m; return np.where(np.isnan(Q), np.nan, np.expm1(m + np.where(d < 0, fl, fu) * d))
F32 = dict(fam); F32["qar"] = tf(fam["qar"], .4, 1.0); L32 = lg(combine(F32)); base = wis_rows(L32, ly)
ed = pd.read_csv(Path(__file__).parents[1] / "data/clean/ed_flu.csv", dtype={"location": str}, parse_dates=["date"])
E = ed.set_index(["location", "date"]).value
V = Vintages(HUB); D = np.full(len(keys), np.nan)
for r in keys.ref.unique():
    last = r - pd.Timedelta(weeks=1); v = V.as_published(last)
    if v is None: continue
    H = v.set_index(["location", "date"]).value
    ix = np.where(keys.ref.values == r)[0]
    for i in ix:
        l = keys.location.values[i]
        try:
            hs = np.array([H.get((l, last - pd.Timedelta(weeks=j)), np.nan) for j in range(5)])
            es = np.array([E.get((l, last - pd.Timedelta(weeks=j)), np.nan) for j in range(5)])
        except Exception: continue
        if np.isnan(hs).any() or np.isnan(es).any() or (es[1:] <= 0).any() or es[0] <= 0: continue
        ratio = np.median(hs[1:] / es[1:]); D[i] = np.log1p(ratio * es[0]) - np.log1p(hs[0])
ok = ~np.isnan(D); print(f"rows with an ER anchor: {ok.mean():.3f}; median |d| small {np.nanmedian(np.abs(D[newest < 100])):.3f} big {np.nanmedian(np.abs(D[newest >= 100])):.3f}")
small = np.nan_to_num(newest, nan=1e9) < 100
def shifted(w):
    w = np.where(ok, w, 0.0); return wis_rows(L32 + (w * np.nan_to_num(D))[:, None], ly)
def evaluate(name, grid, fn):
    cache = {p: fn(p) for p in grid}; out = np.empty(len(y)); ch = []
    for s in SEASONS:
        tr = seas != s; best = min(grid, key=lambda p: cache[p][tr].mean()); out[seas == s] = cache[best][seas == s]; ch.append((s, best))
    d = pd.DataFrame({"ref": keys.ref_s, "n": out, "b": base}).groupby("ref").sum(); rng = np.random.default_rng(0)
    bs = [(lambda x: x.n.sum() / x.b.sum() - 1)(d.sample(len(d), replace=True, random_state=rng.integers(1e9))) for _ in range(2000)]
    lo, hi = np.percentile(bs, [2.5, 97.5]); per = {s: round(float(out[seas == s].sum() / base[seas == s].sum() - 1), 4) for s in SEASONS}
    print(f"{name}: {out.sum()/base.sum()-1:+.2%} CI [{lo:+.2%},{hi:+.2%}] small {out[small].sum()/base[small].sum()-1:+.2%} per-season {per}")
    print("  chosen", ch); print("  KEEP" if hi < 0 and max(per.values()) <= 0 else "  fail")
evaluate("E1", list(itertools.product([0, .25, .5, .75, 1], [50, 100, 200])), lambda p: shifted(np.where(newest < p[1], p[0], 0.0)))
evaluate("E2", [5, 10, 20, 40, 80], lambda n0: shifted(n0 / (n0 + np.nan_to_num(newest, nan=1e9))))
