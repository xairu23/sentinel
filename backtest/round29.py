"""Round 29: recalibration by phase, Vincent/linear-pool mix, exterior trimming (vs v3.4).  python -m backtest.round29"""
import io, contextlib, runpy, itertools, numpy as np, pandas as pd
from scipy import stats, optimize
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_module("backtest.round28", run_name="x")
keys, seas, ly, fam, T, have, w, newest = (g[k] for k in ["keys", "seas", "ly", "fam", "T", "have", "w", "newest"])
g26 = g["g"]["g"]
from backtest.eval_all53 import wis_rows
from backtest.exp_v3 import SEASONS
from backtest.fam_rows import combine
from backtest.round10 import lg
from backtest.wis import QUANTILES as QS
tf = g["tf"]; y = keys.y.values; h = keys.horizon.values
F = dict(fam); F["tree"] = np.where(have[:, None], T, fam["tree"]); F["qar"] = tf(fam["qar"], .4, 1.0)
L34 = lg(combine(F)) + w; base = wis_rows(L34, ly)
# phase from ER share growth as published (same sources as round 26b), else hospital growth as published
fin, corr, pooled, ed_by_asof, V = g26["fin"], g26["corr"], g26["pooled"], g26["ed_by_asof"], g26["V"]
gr = np.full(len(keys), np.nan)
for r in keys.ref.unique():
    last = r - pd.Timedelta(weeks=1); prev = last - pd.Timedelta(weeks=1)
    asof = V.t[V.t.date == last].as_of.min(); Eh = ed_by_asof.get(asof); v = V.as_published(last)
    H = v.set_index(["location", "date"]).value if v is not None else None
    for i in np.where(keys.ref.values == r)[0]:
        l = keys.location.values[i]
        if Eh is not None and (l, last) in Eh.index:
            e0, e1 = Eh.get((l, last), np.nan), Eh.get((l, prev), np.nan)
        else:
            e0, e1 = fin.get((l, last), np.nan) / np.exp(corr.get(l, pooled)), fin.get((l, prev), np.nan)
        if e0 > 0 and e1 > 0: gr[i] = np.log(e0 / e1)
        elif H is not None: gr[i] = np.log1p(H.get((l, last), np.nan)) - np.log1p(H.get((l, prev), np.nan))
phase = np.where(np.nan_to_num(gr) > .1, 2, np.where(np.nan_to_num(gr) < -.1, 0, 1))
print("phase shares (falling, flat, rising):", np.bincount(phase, minlength=3) / len(phase))

def rule(name, new, extra=""):
    d = pd.DataFrame({"ref": keys.ref_s, "n": new, "b": base}).groupby("ref").sum(); rng = np.random.default_rng(0)
    bs = [(lambda x: x.n.sum() / x.b.sum() - 1)(d.sample(len(d), replace=True, random_state=rng.integers(1e9))) for _ in range(2000)]
    lo, hi = np.percentile(bs, [2.5, 97.5]); pr = {s: round(float(new[seas == s].sum() / base[seas == s].sum() - 1), 4) for s in SEASONS}
    print(f"{name}: {new.sum()/base.sum()-1:+.2%} CI [{lo:+.2%},{hi:+.2%}] per-season {pr} {extra}")
    print("  KEEP" if hi < 0 and max(pr.values()) <= 0 else "  fail")

# --- P1: Beta recalibration per (phase x horizon), fitted on the other seasons
def pit(L, yy):
    return np.array([np.interp(v, l, QS, left=QS[0] / 2, right=1 - (1 - QS[-1]) / 2) for l, v in zip(L, yy)])
U = np.clip(pit(L34, ly), 1e-4, 1 - 1e-4)
def qfun(L, t):  # quantile function on the log scale at levels t (rows x levels), linear extrapolation outside
    out = np.empty_like(t)
    for i in range(len(L)):
        l = L[i]; tt = t[i]
        mid = np.interp(tt, QS, l)
        lo_s = (l[1] - l[0]) / (QS[1] - QS[0]); hi_s = (l[-1] - l[-2]) / (QS[-1] - QS[-2])
        mid = np.where(tt < QS[0], l[0] - (QS[0] - tt) * lo_s, mid); mid = np.where(tt > QS[-1], l[-1] + (tt - QS[-1]) * hi_s, mid)
        out[i] = mid
    return np.maximum.accumulate(out, axis=1)
def fit_beta(u):
    nll = lambda p: -stats.beta.logpdf(u, np.exp(p[0]), np.exp(p[1])).sum()
    r = optimize.minimize(nll, [0, 0], method="Nelder-Mead"); return np.exp(r.x)
for nm, half in [("P1", False), ("P1h", True)]:
    new = base.copy(); fits = {}
    for s in SEASONS:
        for ph in range(3):
            for hh in range(4):
                tr = (seas != s) & (phase == ph) & (h == hh); te = (seas == s) & (phase == ph) & (h == hh)
                if te.sum() == 0 or tr.sum() < 50: continue
                a, b = fit_beta(U[tr]); fits[(s, ph, hh)] = (round(a, 2), round(b, 2))
                tg = stats.beta.ppf(QS, a, b).clip(.001, .999)
                if half: tg = .5 * tg + .5 * QS
                Lr = qfun(L34[te], np.tile(tg, (te.sum(), 1)))
                new[te] = wis_rows(Lr, ly[te])
    rule(nm, new); print("  fits (season left out, phase, horizon) -> (a, b):", {k: v for k, v in list(fits.items())[:12]}, "...")

# --- P2: mix of quantile averaging and linear pool, lambda per phase
def lop(Ls):
    # average of CDFs on a grid, inverted at QS; Ls = list of (rows x 23) log-quantile arrays (nan rows allowed)
    out = np.empty_like(Ls[0])
    for i in range(len(Ls[0])):
        ls = [l[i] for l in Ls if not np.isnan(l[i]).any()]
        grid = np.linspace(min(l[0] for l in ls) - .5, max(l[-1] for l in ls) + .5, 400)
        cdf = np.mean([np.interp(grid, l, QS, left=0, right=1) for l in ls], axis=0)
        out[i] = np.interp(QS, cdf + np.arange(len(grid)) * 1e-12, grid)
    return out
Lv = lg(combine(F)); Ll = lop([lg(F["reg"]), lg(F["tree"]), lg(F["qar"])])
cache = {lam: wis_rows((1 - lam) * Lv + lam * Ll + w, ly) for lam in [0, .25, .5, .75, 1]}
new = base.copy(); ch = []
for s in SEASONS:
    for ph in range(3):
        tr = (seas != s) & (phase == ph); te = (seas == s) & (phase == ph)
        best = min(cache, key=lambda lam: cache[lam][tr].mean()); new[te] = cache[best][te]; ch.append((s, ph, best))
rule("P2", new, f"chosen {ch}")

# --- P4: exterior trimming in small places
small = np.nan_to_num(newest, nan=1e9) < 100
S = np.stack([F["reg"], F["tree"], F["qar"]]); ok3 = ~np.isnan(S).any(2).any(0)
srt = np.sort(S, axis=0)
lowside = (srt[1] + srt[2]) / 2; highside = (srt[0] + srt[1]) / 2; allm = np.nanmean(S, 0)
for nm, up_only in [("P4", False), ("P4u", True)]:
    Q = combine(F).copy(); m = small & ok3
    tri = np.where(QS < .5, lowside, np.where(QS > .5, highside, allm))
    if up_only: tri = np.where(QS > .5, highside, allm if False else combine(F))
    Q[m] = np.maximum.accumulate(np.maximum(tri[m], 0), axis=1)
    rule(nm, wis_rows(lg(Q) + w, ly), f"small {wis_rows(lg(Q) + w, ly)[small].sum()/base[small].sum()-1:+.2%}")
