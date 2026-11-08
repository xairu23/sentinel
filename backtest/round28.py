"""Round 28: neighbour-feature trees inside v3.3.  python -m backtest.round28"""
import io, contextlib, runpy, numpy as np, pandas as pd
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_module("backtest.round26c", run_name="x")
keys, seas, ly, D, newest, er, ok = (g[k] for k in ["keys", "seas", "ly", "D", "newest", "er", "ok"])
from backtest.eval_all53 import wis_rows, load_family
from backtest.exp_v3 import SEASONS
from backtest.fam_rows import get, combine
from backtest.round10 import lg
_, _, fam, _ = get()
sp = load_family("gbqspbf"); T = np.full_like(fam["tree"], np.nan)
for i, (r, l, hh) in enumerate(zip(keys.ref, keys.location, keys.horizon)):
    q = sp.get(r, {}).get("gbqspbf", {}).get((l, hh + 1)) if r in sp else None
    if q is not None: T[i] = q
have = ~np.isnan(T).any(1)
print(f"rows with neighbour trees: {have.mean():.3f}; refs {len(sp)}")
def tf(Q, fl, fu):
    L = lg(Q); m = L[:, 11:12]; d = L - m; return np.where(np.isnan(Q), np.nan, np.expm1(m + np.where(d < 0, fl, fu) * d))
w = np.where(ok & (np.nan_to_num(newest, nan=1e12) < 200) & (np.nan_to_num(er) >= .01), .5, 0.0)[:, None] * np.nan_to_num(D)[:, None]
def v33(tree):
    f = dict(fam); f["tree"] = tree; f["qar"] = tf(fam["qar"], .4, 1.0); return wis_rows(lg(combine(f)) + w, ly)
base = v33(fam["tree"]); new = v33(np.where(have[:, None], T, fam["tree"]))
m = have; d = pd.DataFrame({"ref": keys.ref_s[m], "n": new[m], "b": base[m]}).groupby("ref").sum(); rng = np.random.default_rng(0)
bs = [(lambda x: x.n.sum() / x.b.sum() - 1)(d.sample(len(d), replace=True, random_state=rng.integers(1e9))) for _ in range(2000)]
lo, hi = np.percentile(bs, [2.5, 97.5]); pr = {s: round(float(new[m & (seas == s)].sum() / base[m & (seas == s)].sum() - 1), 4) for s in SEASONS}
print(f"N1: {new[m].sum()/base[m].sum()-1:+.2%} CI [{lo:+.2%},{hi:+.2%}] per-season {pr}")
print("  KEEP" if hi < 0 and max(pr.values()) <= 0 else "  fail")
# 50-state count check (rule: not worse by more than 1%)
y = keys.y.values; states = ~keys.location.isin(["US", "11", "72"]).values
def v33Q(tree):
    f = dict(fam); f["tree"] = tree; f["qar"] = tf(fam["qar"], .4, 1.0); return np.expm1(lg(combine(f)) + w)
c0 = wis_rows(np.round(v33Q(fam["tree"])[states]), y[states]); c1 = wis_rows(np.round(v33Q(np.where(have[:, None], T, fam["tree"]))[states]), y[states])
print(f"50-state counts: {c1.sum()/c0.sum()-1:+.2%}", {s: f"{c1[seas[states] == s].sum()/c0[seas[states] == s].sum()-1:+.2%}" for s in SEASONS})
