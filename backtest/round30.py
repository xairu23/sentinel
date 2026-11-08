"""Round 30: QAR with the corrected newest week inside v3.4.  python -m backtest.round30"""
import io, contextlib, runpy, numpy as np, pandas as pd
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_module("backtest.round28", run_name="x")
keys, seas, ly, fam, T, have, w = (g[k] for k in ["keys", "seas", "ly", "fam", "T", "have", "w"])
from backtest.eval_all53 import wis_rows, load_family
from backtest.exp_v3 import SEASONS
from backtest.fam_rows import combine
from backtest.round10 import lg
from backtest.round3 import no_pr
tf = g["tf"]
qb = load_family("qarbf"); Qb = np.full_like(fam["qar"], np.nan)
for i, (r, l, hh) in enumerate(zip(keys.ref, keys.location, keys.horizon)):
    d = qb.get(r)
    if d is None: continue
    q = no_pr(d[[k for k in d if k not in ("season", "ref")][0]]).get((l, hh + 1))
    if q is not None: Qb[i] = q
print("rows with corrected-input QAR:", np.mean(~np.isnan(Qb).any(1)).round(3), "| same rows as QAR:", np.array_equal(np.isnan(Qb).any(1), np.isnan(fam["qar"]).any(1)))
def v34(qar):
    f = dict(fam); f["tree"] = np.where(have[:, None], T, fam["tree"]); f["qar"] = tf(qar, .4, 1.0); return lg(combine(f)) + w
base = wis_rows(v34(fam["qar"]), ly); new = wis_rows(v34(Qb), ly)
d = pd.DataFrame({"ref": keys.ref_s, "n": new, "b": base}).groupby("ref").sum(); rng = np.random.default_rng(0)
bs = [(lambda x: x.n.sum() / x.b.sum() - 1)(d.sample(len(d), replace=True, random_state=rng.integers(1e9))) for _ in range(2000)]
lo, hi = np.percentile(bs, [2.5, 97.5]); pr = {s: round(float(new[seas == s].sum() / base[seas == s].sum() - 1), 4) for s in SEASONS}
y = keys.y.values; st = ~keys.location.isin(["US", "11", "72"]).values
c0 = wis_rows(np.round(np.expm1(v34(fam["qar"])))[st], y[st]); c1 = wis_rows(np.round(np.expm1(v34(Qb)))[st], y[st])
print(f"C1: {new.sum()/base.sum()-1:+.2%} CI [{lo:+.2%},{hi:+.2%}] per-season {pr} | 50-state counts {c1.sum()/c0.sum()-1:+.2%}")
print("  KEEP" if hi < 0 and max(pr.values()) <= 0 else "  fail")
