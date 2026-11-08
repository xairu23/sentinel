"""Round 23 (2026-10-30): lower and upper side of the range scaled separately (see V3_PROTOCOL.md).
    python -m backtest.round23
"""
import numpy as np, pandas as pd, itertools
from backtest.round10 import load, lg
from backtest.eval_all53 import wis_rows, HUB, load_family
from backtest.exp_v3 import SEASONS
from backtest.member_cache import mean_members
from backtest.round3 import no_pr
from backtest.run_backtest import Vintages

keys, Q0, truth = load(); y = keys.y.values; seas = keys.season.values; h = keys.horizon.values
V = Vintages(HUB); newest = np.full(len(keys), np.nan)
for r in keys.ref.unique():
    last = r - pd.Timedelta(weeks=1); v = V.as_published(last)
    if v is None: continue
    m = dict(zip(v[v.date == last].location, v[v.date == last].value))
    ix = np.where(keys.ref.values == r)[0]; newest[ix] = [m.get(l, np.nan) for l in keys.location.values[ix]]
small = np.nan_to_num(newest, nan=1e9) < 100
L = lg(Q0); ly = lg(y); base = wis_rows(L, ly); med = L[:, 11:12]

# descriptive, per family
F = {f: load_family(f) for f in ["mem", "gbqbf", "qar"]}
fam = {}
for name, get in [("regressions", lambda r: mean_members(F["mem"][r]["mem"])), ("trees", lambda r: F["gbqbf"][r]["gbqbf"]),
                  ("QAR", lambda r: no_pr(F["qar"][r]["qar"]))]:
    Q = np.full_like(Q0, np.nan); cache = {}
    for i, (r, l, hh) in enumerate(zip(keys.ref, keys.location, h)):
        if r not in cache: cache[r] = get(r)
        q = cache[r].get((l, hh + 1))
        if q is not None: Q[i] = q
    fam[name] = lg(Q)
early = keys.target.dt.month.isin([10, 11]).values
print("small places (newest < 100):", small.mean().round(3), "of rows; early-season (Oct-Nov targets) small rows:", (small & early).sum())
rows = []
for name, LL in [("v3.1", L)] + list(fam.items()):
    for nm, m in [("small", small), ("small, Oct-Nov", small & early), ("big", ~small)]:
        ok = m & ~np.isnan(LL).any(1)
        rows.append((name, nm, wis_rows(LL[ok], ly[ok]).mean(), (LL[ok, 21] - LL[ok, 1]).mean(),
                     ((ly[ok] >= LL[ok, 1]) & (ly[ok] <= LL[ok, 21])).mean(), (ly[ok] < LL[ok, 1]).mean(), (ly[ok] > LL[ok, 21]).mean(), (LL[ok, 11] - ly[ok]).mean()))
print(pd.DataFrame(rows, columns=["family", "rows", "WIS", "width95", "cover95", "below2.5", "above97.5", "bias"]).round(3).to_string(index=False))

def evaluate(name, grid, fn, by_h=True):
    cache = {p: fn(p) for p in grid}
    out = np.empty(len(y)); ch = []
    for s in SEASONS:
        for hh in (range(4) if by_h else [None]):
            tr = (seas != s) & ((h == hh) if by_h else True); te = (seas == s) & ((h == hh) if by_h else True)
            best = min(grid, key=lambda p: cache[p][tr].mean()); out[te] = cache[best][te]; ch.append((s, hh, best))
    d = pd.DataFrame({"ref": keys.ref_s, "n": out, "b": base}).groupby("ref").sum(); rng = np.random.default_rng(0)
    bs = [(lambda x: x.n.sum() / x.b.sum() - 1)(d.sample(len(d), replace=True, random_state=rng.integers(1e9))) for _ in range(2000)]
    lo, hi = np.percentile(bs, [2.5, 97.5]); per = {s: round(out[seas == s].sum() / base[seas == s].sum() - 1, 4) for s in SEASONS}
    cnt = lambda Lx: None
    print(f"{name}: {out.sum()/base.sum()-1:+.2%} CI [{lo:+.2%},{hi:+.2%}] small {out[small].sum()/base[small].sum()-1:+.2%} "
          f"small Oct-Nov {out[small & early].sum()/base[small & early].sum()-1:+.2%} per-season {per}")
    print("  chosen", ch); print("  KEEP" if hi < 0 and max(per.values()) <= 0 else "  fail")

G = [.7, .8, .9, 1.0, 1.1]; grid = list(itertools.product(G, G))
def tails(fl, fu, mask):
    dev = L - med; f = np.where(dev < 0, fl, fu); f = np.where(mask[:, None], f, 1.0)
    return wis_rows(med + f * dev, ly)
evaluate("T1", grid, lambda p: tails(p[0], p[1], np.ones(len(y), bool)))
evaluate("T2", grid, lambda p: tails(p[0], p[1], small))
