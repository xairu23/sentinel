"""Per-family replay quantiles aligned row by row with backtest.round10.load() (cached)."""
import pickle, numpy as np, pandas as pd
from pathlib import Path
from backtest.round10 import load
from backtest.eval_all53 import load_family, HUB
from backtest.member_cache import mean_members
from backtest.round3 import no_pr
from backtest.run_backtest import Vintages
CACHE = Path(__file__).parent / "cache/fam_rows.pkl"

def get():
    if CACHE.exists():
        return pickle.loads(CACHE.read_bytes())
    keys, Q0, truth = load()
    F = {f: load_family(f) for f in ["mem", "gbqbf", "qar"]}
    get_ = {"reg": lambda r: mean_members(F["mem"][r]["mem"]), "tree": lambda r: F["gbqbf"][r]["gbqbf"],
            "qar": lambda r: no_pr(F["qar"][r]["qar"])}
    fam = {}
    for name, g in get_.items():
        Q = np.full_like(Q0, np.nan); c = {}
        for i, (r, l, hh) in enumerate(zip(keys.ref, keys.location, keys.horizon)):
            if r not in c: c[r] = g(r)
            q = c[r].get((l, hh + 1))
            if q is not None: Q[i] = q
        fam[name] = Q
    V = Vintages(HUB); newest = np.full(len(keys), np.nan)
    for r in keys.ref.unique():
        last = r - pd.Timedelta(weeks=1); v = V.as_published(last)
        if v is None: continue
        m = dict(zip(v[v.date == last].location, v[v.date == last].value))
        ix = np.where(keys.ref.values == r)[0]; newest[ix] = [m.get(l, np.nan) for l in keys.location.values[ix]]
    out = (keys, Q0, fam, newest)
    CACHE.write_bytes(pickle.dumps(out)); return out

def combine(fam):
    """v3.1's family average: mean of the available families' quantiles (count scale), made non-decreasing."""
    S = np.stack([fam["reg"], fam["tree"], fam["qar"]]); n = (~np.isnan(S)).all(2).sum(0)
    Q = np.nansum(np.where(np.isnan(S), 0, S), 0) / n[:, None]
    return np.maximum.accumulate(np.maximum(Q, 0), axis=1)
