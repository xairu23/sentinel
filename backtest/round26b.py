"""Round 26b: E1/E2 with ER as first published (see V3_PROTOCOL.md).  python -m backtest.round26b"""
import numpy as np, pandas as pd, itertools
from pathlib import Path
from backtest.round10 import lg
from backtest.eval_all53 import wis_rows, HUB
from backtest.exp_v3 import SEASONS
from backtest.fam_rows import get, combine
from backtest.run_backtest import Vintages
ROOT = Path(__file__).parents[1]
keys, Q0, fam, newest = get(); y = keys.y.values; seas = keys.season.values; h = keys.horizon.values; ly = lg(y)
def tf(Q, fl, fu):
    L = lg(Q); m = L[:, 11:12]; d = L - m; return np.where(np.isnan(Q), np.nan, np.expm1(m + np.where(d < 0, fl, fu) * d))
F32 = dict(fam); F32["qar"] = tf(fam["qar"], .4, 1.0); L32 = lg(combine(F32)); base = wis_rows(L32, ly)
fin = pd.read_csv(ROOT / "data/clean/ed_flu.csv", dtype={"location": str}, parse_dates=["date"]).set_index(["location", "date"]).value
t = pd.read_csv(ROOT / "data/raw/time-series.csv", dtype={"location": str}, parse_dates=["as_of", "target_end_date"])
ev = t[t.target == "wk inc flu prop ed visits"].rename(columns={"target_end_date": "date", "observation": "value"})
mx = ev.groupby("as_of").date.max(); ev = ev.merge(mx.rename("maxd"), left_on="as_of", right_index=True)
first = ev[ev.date == ev.maxd].sort_values("as_of").groupby(["location", "date"]).first().value
rv = pd.concat([first.rename("f"), fin.rename("F")], axis=1).dropna(); rv = rv[rv.f > 0]; rv["lr"] = np.log(rv.F / rv.f)
pooled = rv.lr.median(); per = rv.groupby(level=0).lr.agg(["median", "size"])
corr = {l: (m if n >= 5 else pooled) for l, (m, n) in per.iterrows()}
ed_by_asof = {a: g.set_index(["location", "date"]).value for a, g in ev.groupby("as_of")}
V = Vintages(HUB); D = np.full(len(keys), np.nan); src = np.zeros(len(keys), int)
for r in keys.ref.unique():
    last = r - pd.Timedelta(weeks=1); v = V.as_published(last)
    if v is None: continue
    asof = V.t[(V.t.date == last)].as_of.min()       # the hospital version used (first with newest week = last)
    H = v.set_index(["location", "date"]).value
    Eh = ed_by_asof.get(asof)
    for i in np.where(keys.ref.values == r)[0]:
        l = keys.location.values[i]
        hs = np.array([H.get((l, last - pd.Timedelta(weeks=j)), np.nan) for j in range(5)])
        if Eh is not None and (l, last) in Eh.index:
            es = np.array([Eh.get((l, last - pd.Timedelta(weeks=j)), np.nan) for j in range(5)]); src[i] = 1
        else:
            es = np.array([fin.get((l, last - pd.Timedelta(weeks=j)), np.nan) for j in range(5)])
            es[0] = es[0] / np.exp(corr.get(l, pooled)); src[i] = 2
        if np.isnan(hs).any() or np.isnan(es).any() or (es <= 0).any(): continue
        D[i] = np.log1p(np.median(hs[1:] / es[1:]) * es[0]) - np.log1p(hs[0])
ok = ~np.isnan(D); small = np.nan_to_num(newest, nan=1e9) < 100
print(f"rows with anchor {ok.mean():.3f}; ER as archived {np.mean(src == 1):.3f}, final adjusted {np.mean(src == 2):.3f}; mean d {np.nanmean(D):+.3f}")
def shifted(w):
    w = np.where(ok, w, 0.0); return wis_rows(L32 + (w * np.nan_to_num(D))[:, None], ly)
def evaluate(name, grid, fn):
    cache = {p: fn(p) for p in grid}; out = np.empty(len(y)); ch = []
    for s in SEASONS:
        tr = seas != s; best = min(grid, key=lambda p: cache[p][tr].mean()); out[seas == s] = cache[best][seas == s]; ch.append((s, best))
    d = pd.DataFrame({"ref": keys.ref_s, "n": out, "b": base}).groupby("ref").sum(); rng = np.random.default_rng(0)
    bs = [(lambda x: x.n.sum() / x.b.sum() - 1)(d.sample(len(d), replace=True, random_state=rng.integers(1e9))) for _ in range(2000)]
    lo, hi = np.percentile(bs, [2.5, 97.5]); pr = {s: round(float(out[seas == s].sum() / base[seas == s].sum() - 1), 4) for s in SEASONS}
    a = src == 1
    print(f"{name}: {out.sum()/base.sum()-1:+.2%} CI [{lo:+.2%},{hi:+.2%}] small {out[small].sum()/base[small].sum()-1:+.2%} per-season {pr}")
    print(f"  rows with archived ER only: {out[a].sum()/base[a].sum()-1:+.2%} (n={a.sum()})")
    print("  chosen", ch); print("  KEEP" if hi < 0 and max(pr.values()) <= 0 else "  fail")
    return cache
c1 = evaluate("E1", list(itertools.product([0, .25, .5, .75, 1], [50, 100, 200])), lambda p: shifted(np.where(newest < p[1], p[0], 0.0)))
evaluate("E2", [5, 10, 20, 40, 80], lambda n0: shifted(n0 / (n0 + np.nan_to_num(newest, nan=1e9))))
fx = c1[(.5, 200)]
print("fixed a=0.5, T=200:", f"{fx.sum()/base.sum()-1:+.2%}", {s: f"{fx[seas == s].sum()/base[seas == s].sum()-1:+.2%}" for s in SEASONS},
      "archived-ER rows", f"{fx[src == 1].sum()/base[src == 1].sum()-1:+.2%}")
