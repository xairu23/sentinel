"""Round 24 (2026-10-30): fix the widest family (see V3_PROTOCOL.md).   python -m backtest.round24"""
import numpy as np, pandas as pd, itertools
from backtest.round10 import lg
from backtest.eval_all53 import wis_rows
from backtest.exp_v3 import SEASONS
from backtest.fam_rows import get, combine

keys, Q0, fam, newest = get(); y = keys.y.values; seas = keys.season.values; h = keys.horizon.values
small = np.nan_to_num(newest, nan=1e9) < 100; early = keys.target.dt.month.isin([10, 11]).values
ly = lg(y); Lc = lg(combine(fam)); base = wis_rows(lg(Q0), ly)
print("check: recombined families == v3.1 replay:", np.allclose(Lc, lg(Q0), atol=1e-6), "max diff", np.abs(Lc - lg(Q0)).max())

def evaluate(name, grid, fn, by_h=True):
    cache = {p: fn(p) for p in grid}
    out = np.empty(len(y)); ch = []
    for s in SEASONS:
        for hh in (range(4) if by_h else [None]):
            tr = (seas != s) & ((h == hh) if by_h else True); te = (seas == s) & ((h == hh) if by_h else True)
            best = min(grid, key=lambda p: cache[p][tr].mean()); out[te] = cache[best][te]; ch.append((s, hh, best))
    d = pd.DataFrame({"ref": keys.ref_s, "n": out, "b": base}).groupby("ref").sum(); rng = np.random.default_rng(0)
    bs = [(lambda x: x.n.sum() / x.b.sum() - 1)(d.sample(len(d), replace=True, random_state=rng.integers(1e9))) for _ in range(2000)]
    lo, hi = np.percentile(bs, [2.5, 97.5]); per = {s: round(float(out[seas == s].sum() / base[seas == s].sum() - 1), 4) for s in SEASONS}
    print(f"{name}: {out.sum()/base.sum()-1:+.2%} CI [{lo:+.2%},{hi:+.2%}] small {out[small].sum()/base[small].sum()-1:+.2%} "
          f"small Oct-Nov {out[small & early].sum()/base[small & early].sum()-1:+.2%} per-season {per}")
    print("  chosen", sorted(set(c[2] for c in ch)), ch if len(set(c[2] for c in ch)) > 1 else "")
    print("  KEEP" if hi < 0 and max(per.values()) <= 0 else "  fail")
    return out

med = Lc[:, 11:12]
def tails(fl, fu, mask, L=Lc):
    m = L[:, 11:12]; dev = L - m; f = np.where(dev < 0, fl, fu); f = np.where(mask[:, None], f, 1.0)
    return med_ok(m + f * dev)
med_ok = lambda L: L
evaluate("T2b", list(itertools.product([.4, .5, .6, .7, .8, .9, 1], [.8, .9, 1, 1.1, 1.2])),
         lambda p: wis_rows(tails(p[0], p[1], small), ly))

Lq = lg(fam["qar"]); mq = Lq[:, 11:12]
def with_qar(Lq_new):
    f = dict(fam); f["qar"] = np.where(np.isnan(fam["qar"]), np.nan, np.expm1(Lq_new)); return wis_rows(lg(combine(f)), ly)
evaluate("Q1", [.4, .5, .6, .7, .8, .9, 1], lambda a: with_qar(mq + a * (Lq - mq)))
evaluate("Q2", list(itertools.product([.4, .6, .8, 1], [.4, .6, .8, 1])),
         lambda p: with_qar(mq + np.where(Lq - mq < 0, p[0], p[1]) * (Lq - mq)))
Lr, Lt = lg(fam["reg"]), lg(fam["tree"])
dev_other = ((Lr - Lr[:, 11:12]) + (Lt - Lt[:, 11:12])) / 2
evaluate("Q3", [0], lambda _: with_qar(np.maximum.accumulate(mq + dev_other, axis=1)), by_h=False)
