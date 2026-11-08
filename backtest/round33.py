"""Round 33: ER visits inside the QAR, within v3.6 (vs v3.6).  python -m backtest.round33"""
import io, contextlib, runpy, numpy as np, pandas as pd
with contextlib.redirect_stdout(io.StringIO()):
    g = runpy.run_module("backtest.round32", run_name="x")
keys, seas, ly, fam, w, tf, rows, compare = (g[k] for k in ["keys", "seas", "ly", "fam", "w", "tf", "rows", "compare"])
from backtest.fam_rows import combine
from backtest.round10 import lg
E, N = g["E"], g["N"]; Q = rows("qared")
tree = np.where(~np.isnan(E).any(1)[:, None], E, N)
qok = ~np.isnan(Q).any(1); print(f"rows with ER-QAR: {qok.mean():.3f}")
g["hv"][:] = qok  # compare() scores these rows


def v36(qar):
    f = dict(fam); f["tree"] = tree; f["qar"] = tf(qar, .4, 1.0); return lg(combine(f)) + w


ok = compare(v36(fam["qar"]), v36(np.where(qok[:, None], Q, fam["qar"])), "Q1 (v3.6 + ER QAR vs v3.6)")
print("  KEEP" if ok else "  fail")
