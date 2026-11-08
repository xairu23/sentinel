"""Round 20 (2026-10-21): TEMPORARY early-season fix requested by the team (not a protocol KEEP).
Shift down by c (log scale) the forecasts of places whose as-published newest week is < 100, for refs in
October-November only. Report replay per season and the two live weeks."""
import numpy as np, pandas as pd, glob
from backtest.round10 import load, lg, mono
from backtest.eval_all53 import wis_rows, HUB
from backtest.exp_v3 import SEASONS
from backtest.run_backtest import Vintages
keys, Q0, truth = load(); y = keys.y.values; seas = keys.season.values; L = lg(Q0)
V = Vintages(HUB); pub = {}
for r in keys.ref.unique():
    last = r - pd.Timedelta(weeks=1); v = V.as_published(last)
    if v is not None:
        v = v[v.date == last]; pub.update({(str(r.date()), l): x for l, x in zip(v.location, v.value)})
lvl = np.array([pub.get((r, l), np.nan) for r, l in zip(keys.ref_s, keys.location)], float)
m = (lvl < 100) & keys.ref.dt.month.isin([10, 11]).values
base = wis_rows(L, lg(y))
for c in [.03, .05, .08, .1]:
    new = wis_rows(L - c * m[:, None], lg(y))
    per = {s: f"{new[(seas==s)&m].sum()/base[(seas==s)&m].sum()-1:+.1%}" for s in SEASONS}
    print(f"replay c={c}: affected rows {per}  whole-season {new.sum()/base.sum()-1:+.2%}")
# live
t = pd.read_csv('/tmp/tgt.csv', dtype={'location': str}); tv = {(r.date, r.location): r.value for r in t.itertuples()}
ts = pd.read_csv('/tmp/ts.csv', dtype={'location': str}); ts = ts[ts.target == 'wk inc flu hosp']
for ref, asof in [('2026-10-10', '2026-10-07'), ('2026-10-17', '2026-10-14')]:
    last = str((pd.Timestamp(ref) - pd.Timedelta(weeks=1)).date())
    pv = ts[(ts.as_of == asof) & (ts.target_end_date == last)].set_index('location').observation.astype(float)
    d = pd.read_csv(f'submission/output/{ref}-Sentinel-Ens.csv', dtype={'location': str, 'output_type_id': str})
    d = d[d.target == 'wk inc flu hosp']
    for c in [0, .05, .1]:
        s = []
        for (loc, h), g in d.groupby(['location', 'horizon']):
            te = str((pd.Timestamp(ref) + pd.Timedelta(weeks=int(h))).date()); yy = tv.get((te, loc))
            if yy is None or loc == 'US': continue
            q = g.output_type_id.astype(float).values; v = np.log1p(g.value.astype(float).values)
            if pv.get(loc, 1e9) < 100: v = v - c
            s.append(np.mean(2 * (((np.log1p(yy) < v) - q) * (v - np.log1p(yy)))))
        print(f"live ref {ref} c={c}: mean WIS {np.mean(s):.4f} (n={len(s)})")
