"""Round 31: national + season-position tree features inside v3.4 (vs v3.4).  python -m backtest.round31"""
import io, contextlib, runpy, numpy as np, pandas as pd
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_module("backtest.round28", run_name="x")
keys, seas, ly, fam, T, have, w = (g[k] for k in ["keys", "seas", "ly", "fam", "T", "have", "w"])
from backtest.eval_all53 import wis_rows, load_family
from backtest.exp_v3 import SEASONS
from backtest.fam_rows import combine
from backtest.round10 import lg
tf = g["tf"]
sp = load_family("gbqspnbf"); N = np.full_like(fam["tree"], np.nan)
for i, (r, l, hh) in enumerate(zip(keys.ref, keys.location, keys.horizon)):
    q = sp.get(r, {}).get("gbqspnbf", {}).get((l, hh + 1)) if r in sp else None
    if q is not None: N[i] = q
hv = ~np.isnan(N).any(1); print(f"rows with round-31 trees: {hv.mean():.3f}; refs {len(sp)}")
T34 = np.where(have[:, None], T, fam["tree"])
def v34(tree):
    f = dict(fam); f["tree"] = tree; f["qar"] = tf(fam["qar"], .4, 1.0); return lg(combine(f)) + w
base = wis_rows(v34(T34), ly); new = wis_rows(v34(np.where(hv[:, None], N, T34)), ly)
m = hv; d = pd.DataFrame({"ref": keys.ref_s[m], "n": new[m], "b": base[m]}).groupby("ref").sum(); rng = np.random.default_rng(0)
bs = [(lambda x: x.n.sum() / x.b.sum() - 1)(d.sample(len(d), replace=True, random_state=rng.integers(1e9))) for _ in range(2000)]
lo, hi = np.percentile(bs, [2.5, 97.5]); pr = {s: round(float(new[m & (seas == s)].sum() / base[m & (seas == s)].sum() - 1), 4) for s in SEASONS}
y = keys.y.values; st = ~keys.location.isin(["US", "11", "72"]).values & m
c0 = wis_rows(np.round(np.expm1(v34(T34)))[st], y[st]); c1 = wis_rows(np.round(np.expm1(v34(np.where(hv[:, None], N, T34))))[st], y[st])
small = np.nan_to_num(g["newest"], nan=1e9) < 100
print(f"N2: {new[m].sum()/base[m].sum()-1:+.2%} CI [{lo:+.2%},{hi:+.2%}] per-season {pr} | small {new[m & small].sum()/base[m & small].sum()-1:+.2%} | 50-state counts {c1.sum()/c0.sum()-1:+.2%}")
print("  KEEP" if hi < 0 and max(pr.values()) <= 0 else "  fail")
