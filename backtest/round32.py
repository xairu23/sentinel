"""Round 32: ER visits inside v3.5's trees (vs v3.5).  python -m backtest.round32"""
import io, contextlib, runpy, numpy as np, pandas as pd
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_module("backtest.round28", run_name="x")
keys, seas, ly, fam, T, have, w = (g[k] for k in ["keys", "seas", "ly", "fam", "T", "have", "w"])
from backtest.eval_all53 import wis_rows, load_family
from backtest.exp_v3 import SEASONS
from backtest.fam_rows import combine
from backtest.round10 import lg
tf = g["tf"]


def rows(name):
    sp = load_family(name); out = np.full_like(fam["tree"], np.nan)
    for i, (r, l, hh) in enumerate(zip(keys.ref, keys.location, keys.horizon)):
        q = sp.get(r, {}).get(name, {}).get((l, hh + 1)) if r in sp else None
        if q is not None: out[i] = q
    return out


N = rows("gbqspnbf"); E = rows("gbqedbf")
hv = ~np.isnan(N).any(1) & ~np.isnan(E).any(1); print(f"rows with both: {hv.mean():.3f}")


def ens(tree, anchor=True):
    f = dict(fam); f["tree"] = tree; f["qar"] = tf(fam["qar"], .4, 1.0); return lg(combine(f)) + (w if anchor else 0)


def compare(lb, ln, label):
    base, new = wis_rows(lb, ly), wis_rows(ln, ly); m = hv
    d = pd.DataFrame({"ref": keys.ref_s[m], "n": new[m], "b": base[m]}).groupby("ref").sum(); rng = np.random.default_rng(0)
    bs = [(lambda x: x.n.sum() / x.b.sum() - 1)(d.sample(len(d), replace=True, random_state=rng.integers(1e9))) for _ in range(2000)]
    lo, hi = np.percentile(bs, [2.5, 97.5]); pr = {s: round(float(new[m & (seas == s)].sum() / base[m & (seas == s)].sum() - 1), 4) for s in SEASONS}
    y = keys.y.values; st = ~keys.location.isin(["US", "11", "72"]).values & m
    c0 = wis_rows(np.round(np.expm1(lb))[st], y[st]); c1 = wis_rows(np.round(np.expm1(ln))[st], y[st])
    small = np.nan_to_num(g["newest"], nan=1e9) < 100
    mon = pd.to_datetime(keys.ref_s.values).month
    bym = {mm: f"{new[m & (mon == mm)].sum() / base[m & (mon == mm)].sum() - 1:+.1%}" for mm in [10, 11, 12, 1, 2, 3, 4, 5]}
    print(f"{label}: {new[m].sum()/base[m].sum()-1:+.2%} CI [{lo:+.2%},{hi:+.2%}] per-season {pr} | small "
          f"{new[m & small].sum()/base[m & small].sum()-1:+.2%} | 50-state counts {c1.sum()/c0.sum()-1:+.2%}\n  by month {bym}")
    return hi < 0 and max(pr.values()) <= 0 and c1.sum() / c0.sum() - 1 <= 0.01


ok = compare(ens(N), ens(np.where(hv[:, None], E, N)), "E1 (v3.5 + ER trees vs v3.5)")
print("  KEEP" if ok else "  fail")
# secondary, not for the decision: does the ER anchor still add anything once the trees see ER visits?
compare(ens(np.where(hv[:, None], E, N), anchor=False), ens(np.where(hv[:, None], E, N)), "secondary: anchor on vs off, with ER trees")
