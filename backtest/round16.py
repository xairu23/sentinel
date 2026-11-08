"""Round 16 (2026-10-21): lagged-ensemble member. Idea: the CDC FluSight-ensemble's forecast published
LAST week is public before our deadline. For our horizon h it gives a 1-week-older forecast of the same
target week (its horizon h+1). It pools ~50 teams, so its errors should be only partly correlated with
ours. Blend: Vincent average in log space, weight w per horizon (h0-h2; h3 has no lagged counterpart),
chosen leave-one-season-out. Same keep rule as rounds 3-15.
"""
import numpy as np, pandas as pd
from backtest.round10 import load, lg, mono
from backtest.eval_all53 import HUB, wis_rows
from backtest.exp_v3 import SEASONS
from backtest.leaderboard import load_model

keys, Q0, truth = load(); y = keys.y.values; seas = keys.season.values; h = keys.horizon.values
prev = (keys.ref - pd.Timedelta(weeks=1)).dt.strftime("%Y-%m-%d")
want = pd.DataFrame({"ref": prev, "location": keys.location, "horizon": h + 1})
E = load_model(HUB / "model-output/FluSight-ensemble", want)
E.index = E.index.set_levels([E.index.levels[0].astype(str).str[:10], E.index.levels[1], E.index.levels[2]])
lagE = E.reindex(pd.MultiIndex.from_frame(want.astype({"horizon": int}))).values.astype(float)
has = ~np.isnan(lagE).any(1)
print("rows with lagged ensemble:", has.mean().round(3), "by horizon", pd.Series(has).groupby(h).mean().round(2).to_dict())
L0 = lg(Q0); LE = lg(np.nan_to_num(lagE))
base = wis_rows(L0, lg(y))
print("standalone on shared rows: ours %.4f  lagged-ens %.4f" % (base[has].mean(), wis_rows(LE[has], lg(y[has])).mean()))
W = [0, .15, .3, .45, .6]
cache = {w: np.where(has, wis_rows(np.where(has[:, None], (1 - w) * L0 + w * LE, L0), lg(y)), base) for w in W}
out = base.copy(); chosen = []
for s in SEASONS:
    for hh in [0, 1, 2]:
        tr = (seas != s) & (h == hh); te = (seas == s) & (h == hh)
        best = min(W, key=lambda w: cache[w][tr].mean()); out[te] = cache[best][te]; chosen.append((s, hh, best))
d = pd.DataFrame({"ref": keys.ref_s, "n": out, "b": base}).groupby("ref").sum()
rng = np.random.default_rng(0)
bs = [(lambda x: x.n.sum() / x.b.sum() - 1)(d.sample(len(d), replace=True, random_state=rng.integers(1e9))) for _ in range(2000)]
lo, hi = np.percentile(bs, [2.5, 97.5]); per = {s: round(out[seas == s].sum() / base[seas == s].sum() - 1, 4) for s in SEASONS}
byh = {hh: round(out[h == hh].sum() / base[h == hh].sum() - 1, 4) for hh in range(4)}
print(f"L: {out.sum()/base.sum()-1:+.2%} CI [{lo:+.2%},{hi:+.2%}] per-season {per} per-horizon {byh}")
print("chosen", chosen)
print("KEEP" if hi < 0 and max(per.values()) <= 0 else "fail")
