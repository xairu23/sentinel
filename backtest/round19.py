"""Round 19 (2026-10-21): family weights by place size. Live: in big states we match the top teams
(WIS 0.124 vs 0.119-0.139); the whole gap is in places under 100/week (0.192 vs ~0.17), where our median
sits ~10% high (+0.10 log vs ~0 to +0.06 for top teams) and our spread is ~10% wider. Hypothesis: one
family (likely the regressions, whose mean-reversion term pushes low early-season states up) drives it.
Test: separate Vincent weights for (regressions, trees, QAR) in small vs big places, LOSO.
"""
import numpy as np, pandas as pd, itertools
from backtest.eval_all53 import load_family, wis_rows, HUB
from backtest.exp_v3 import SEASONS
from backtest.member_cache import mean_members
from backtest.round3 import no_pr
from backtest.run_backtest import final_hosp, Vintages
lg = lambda a: np.log1p(np.maximum(a, 0))
F = {f: load_family(f) for f in ["mem", "gbqbf", "qar"]}
refs = sorted(set.intersection(*[set(v) for v in F.values()]))
tr = final_hosp().set_index(["location", "date"])["value"]
V = Vintages(HUB)
rows, A, B, C = [], [], [], []
for r in refs:
    fam = [mean_members(F["mem"][r]["mem"]), F["gbqbf"][r]["gbqbf"], no_pr(F["qar"][r]["qar"])]
    last = r - pd.Timedelta(weeks=1); v = V.as_published(last); v = v[v.date == last].set_index("location").value
    for key in fam[0]:
        if any(key not in f for f in fam): continue
        loc, k = key; te = last + pd.Timedelta(weeks=k); y = tr.get((loc, te))
        if y is None or np.isnan(y): continue
        rows.append((F["mem"][r]["season"], str(r.date()), loc, k - 1, float(y), float(v.get(loc, np.nan)), r.month))
        A.append(fam[0][key]); B.append(fam[1][key]); C.append(fam[2][key])
K = pd.DataFrame(rows, columns=["season", "ref", "loc", "h", "y", "lvl", "mon"])
A, B, C = lg(np.array(A)), lg(np.array(B)), lg(np.array(C)); y = lg(K.y.values); small = (K.lvl < 100).values; seas = K.season.values
for n, M in [("regressions", A), ("trees", B), ("QAR", C)]:
    e = M[:, 11] - y
    print(f"{n:12s} small: WIS {wis_rows(M[small], y[small]).mean():.4f} bias {np.median(e[small]):+.3f} | big: WIS {wis_rows(M[~small], y[~small]).mean():.4f} bias {np.median(e[~small]):+.3f}")
eq = (A + B + C) / 3; base = wis_rows(eq, y)
grid = [w for w in itertools.product(np.arange(0, 1.01, .1), repeat=2) if sum(w) <= 1.0001]
def mix(w): return w[0] * A + w[1] * B + (1 - w[0] - w[1]) * C
cache = {w: wis_rows(mix(w), y) for w in grid}
out = base.copy(); ch = []
for s in SEASONS:
    for grp in [True, False]:
        trm = (seas != s) & (small == grp); te = (seas == s) & (small == grp)
        best = min(grid, key=lambda w: cache[w][trm].mean()); out[te] = cache[best][te]; ch.append((s, grp, tuple(round(x, 1) for x in best)))
d = pd.DataFrame({"ref": K.ref, "n": out, "b": base}).groupby("ref").sum(); rng = np.random.default_rng(0)
bs = [(lambda x: x.n.sum() / x.b.sum() - 1)(d.sample(len(d), replace=True, random_state=rng.integers(1e9))) for _ in range(2000)]
lo, hi = np.percentile(bs, [2.5, 97.5]); per = {s: round(out[seas == s].sum() / base[seas == s].sum() - 1, 4) for s in SEASONS}
print(f"W-size: {out.sum()/base.sum()-1:+.2%} CI [{lo:+.2%},{hi:+.2%}] small {out[small].sum()/base[small].sum()-1:+.2%} per-season {per}")
print("chosen (season, small?, (w_regr, w_trees)):", ch); print("KEEP" if hi < 0 and max(per.values()) <= 0 else "fail")
