"""Round 21 (2026-10-21): average the three families on the LOG scale instead of the count scale.
Reason: an arithmetic mean of quantiles is pulled toward the highest member; when members disagree in
relative terms (small places), that pushes the median up. Scored on log scale, a log-scale average is the
natural combination. No tuning: one fixed rule. Same keep rule as before."""
import numpy as np, pandas as pd
from backtest.eval_all53 import load_family, wis_rows, HUB
from backtest.exp_v3 import SEASONS, quantile_average
from backtest.member_cache import mean_members
from backtest.round3 import no_pr
from backtest.run_backtest import final_hosp, Vintages
lg = lambda a: np.log1p(np.maximum(a, 0))
F = {f: load_family(f) for f in ["mem", "gbqbf", "qar"]}
refs = sorted(set.intersection(*[set(v) for v in F.values()]))
tr = final_hosp().set_index(["location", "date"])["value"]; V = Vintages(HUB)
rows, Lv, Lg = [], [], []
for r in refs:
    fam = [mean_members(F["mem"][r]["mem"]), F["gbqbf"][r]["gbqbf"], no_pr(F["qar"][r]["qar"])]
    va = quantile_average(fam)
    last = r - pd.Timedelta(weeks=1); v = V.as_published(last); v = v[v.date == last].set_index("location").value
    for key, q in va.items():
        if any(key not in f for f in fam): continue
        loc, k = key; y = tr.get((loc, last + pd.Timedelta(weeks=k)))
        if y is None or np.isnan(y): continue
        g = np.maximum.accumulate(np.mean([lg(f[key]) for f in fam], 0))
        rows.append((F["mem"][r]["season"], str(r.date()), loc, float(y), float(v.get(loc, np.nan)), r.month))
        Lv.append(lg(q)); Lg.append(g)
K = pd.DataFrame(rows, columns=["season", "ref", "loc", "y", "lvl", "mon"]); y = lg(K.y.values)
base = wis_rows(np.array(Lv), y); new = wis_rows(np.array(Lg), y); seas = K.season.values
small = (K.lvl < 100).values; early = K.mon.isin([10, 11]).values
d = pd.DataFrame({"ref": K.ref, "n": new, "b": base}).groupby("ref").sum(); rng = np.random.default_rng(0)
bs = [(lambda x: x.n.sum() / x.b.sum() - 1)(d.sample(len(d), replace=True, random_state=rng.integers(1e9))) for _ in range(2000)]
lo, hi = np.percentile(bs, [2.5, 97.5]); f = lambda m: f"{new[m].sum()/base[m].sum()-1:+.2%}"
per = {s: f(seas == s) for s in SEASONS}
print(f"LOG-AVG: {f(np.ones(len(y),bool))} CI [{lo:+.2%},{hi:+.2%}] per-season {per} small {f(small)} small-early {f(small&early)} big {f(~small)}")
print("median log bias small-early: count-avg %+.3f log-avg %+.3f" % (np.median(np.array(Lv)[small&early,11]-y[small&early]), np.median(np.array(Lg)[small&early,11]-y[small&early])))
worse = any(new[seas == s].sum() > base[seas == s].sum() for s in SEASONS)
print("KEEP" if hi < 0 and not worse else "fail")
