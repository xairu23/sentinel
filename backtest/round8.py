"""Round 8 (2026-10-14): ideas from studying the consistently best FluSight teams of 2025-26.
Pre-registered before running; same rule as rounds 3-7 (95% CI below 0, no season worse,
50-state counts not worse by more than 1%), compared with v3.1, all 53 places, log WIS.

  A  average of v2 and v3.1 (top teams are ensembles of diverse models)
  A2 median instead of mean across the three v3.1 families
  D  v3.1 families plus two extra members already cached (QAR-weighted, AR6), equal weights
  C  v3.1 with a per-horizon spread factor (log scale), chosen leave-one-season-out
  AC A + per-horizon spread factor
    python -m backtest.round8
"""
from __future__ import annotations
import numpy as np, pandas as pd
from backtest.eval_all53 import HUB, load_family, wis_rows
from backtest.exp_v3 import SEASONS, blend, quantile_average
from backtest.leaderboard import load_model
from backtest.member_cache import mean_members
from backtest.round3 import no_pr
from backtest.run_backtest import final_hosp

lg = lambda a: np.log1p(np.maximum(a, 0))
GRID = np.round(np.arange(0.7, 1.41, 0.05), 2)


def main():
    F = {f: load_family(f) for f in ["mem", "gbq", "gbqbf", "qar", "qarw", "ar6"]}
    refs = sorted(set.intersection(*[set(v) for v in F.values()]))
    season = {r: F["mem"][r]["season"] for r in refs}
    lin = {r: mean_members(F["mem"][r]["mem"]) for r in refs}
    g = {r: F["gbq"][r]["gbq"] for r in refs}
    gb = {r: F["gbqbf"][r]["gbqbf"] for r in refs}
    q = {r: no_pr(F["qar"][r]["qar"]) for r in refs}
    v31 = {r: quantile_average([lin[r], gb[r], q[r]]) for r in refs}
    v2 = {r: blend(lin[r], g[r], 0.33) for r in refs}
    models = {
        "v3.1": v31, "v2": v2,
        "A": {r: quantile_average([v2[r], v31[r]]) for r in refs},
        "A2": {r: quantile_average([lin[r], gb[r], q[r]], how="median") for r in refs},
        "D": {r: quantile_average([lin[r], gb[r], q[r], no_pr(F["qarw"][r]["qarw"]), F["ar6"][r]["ar6"]]) for r in refs},
    }
    truth = final_hosp().set_index(["location", "date"])["value"]
    rows = []
    for r in refs:
        last = r - pd.Timedelta(weeks=1)
        for (loc, k) in v31[r]:
            y = truth.get((loc, last + pd.Timedelta(weeks=k)))
            if y is not None and not np.isnan(y):
                rows.append((season[r], r, str(r.date()), loc, k - 1, float(y)))
    keys = pd.DataFrame(rows, columns=["season", "ref", "ref_s", "location", "horizon", "y"])
    idx = pd.MultiIndex.from_frame(keys[["ref_s", "location", "horizon"]])
    w = load_model(HUB / "model-output/FluSight-baseline", keys.rename(columns={"ref_s": "ref"})[["ref", "location", "horizon"]])
    w.index = w.index.set_names(["ref_s", "location", "horizon"])
    base = w.reindex(idx).values.astype(float)
    ok = ~np.isnan(base).any(axis=1)
    keys, base, y = keys[ok].reset_index(drop=True), base[ok], keys["y"].values[ok]
    Wb = wis_rows(lg(base), lg(y))
    Q = {m: np.vstack([p[r][(l, h + 1)] for r, l, h in zip(keys.ref, keys.location, keys.horizon)]) for m, p in models.items()}

    def spread(Qm, s_by_h):
        L = lg(Qm); med = L[:, [11]]
        s = np.array([s_by_h[h] for h in keys.horizon])[:, None]
        return np.maximum.accumulate(np.expm1(med + s * (L - med)), axis=1)

    def loso(Qm):
        out = Qm.copy()
        for s in SEASONS:
            tr = (keys.season != s).values; te = ~tr
            best = {}
            for h in range(4):
                m = tr & (keys.horizon == h).values
                sc = [wis_rows(lg(spread(Qm[m], {h: f})), lg(y[m])).sum() if False else
                      wis_rows(lg(np.maximum.accumulate(np.expm1(lg(Qm[m])[:, [11]] + f * (lg(Qm[m]) - lg(Qm[m])[:, [11]])), axis=1)), lg(y[m])).sum()
                      for f in GRID]
                best[h] = GRID[int(np.argmin(sc))]
            print(f"  spread factors fitted without {s}: {best}")
            out[te] = spread(Qm[te], best)[:,] if False else np.vstack([spread(Qm[te][i:i+1], best) if False else Qm[te][i] for i in range(0)]) if False else out[te]
            sub = keys[te]
            out[te] = spread_sub(Qm[te], best, sub.horizon.values)
        return out

    def spread_sub(Qm, best, hs):
        L = lg(Qm); med = L[:, [11]]
        s = np.array([best[h] for h in hs])[:, None]
        return np.maximum.accumulate(np.expm1(med + s * (L - med)), axis=1)

    print("C (v3.1 + spread):"); Q["C"] = loso(Q["v3.1"])
    print("AC (A + spread):"); Q["AC"] = loso(Q["A"])
    Q = {m: np.round(v) for m, v in Q.items()}
    W = {m: wis_rows(lg(v), lg(y)) for m, v in Q.items()}
    Cn = {m: wis_rows(v, y) for m, v in Q.items()}
    wk = keys["ref"].values; weeks = np.unique(wk)
    A0 = pd.Series(W["v3.1"]).groupby(wk).sum().reindex(weeks).values
    ii = np.random.default_rng(0).integers(0, len(weeks), size=(2000, len(weeks)))
    states = ~keys["location"].isin(["US", "11", "72"]).values
    print("\n| model | vs baseline | change vs v3.1 (95% CI) | " + " | ".join(SEASONS) + " | counts 50 states | rule |")
    print("|---|---|---|" + "---|" * len(SEASONS) + "---|---|")
    for m in Q:
        B = pd.Series(W[m]).groupby(wk).sum().reindex(weeks).values
        lo, hi = np.percentile((B[ii].sum(1) / A0[ii].sum(1) - 1) * 100, [2.5, 97.5])
        ps = {s: (W[m][(keys.season == s).values].sum() / W["v3.1"][(keys.season == s).values].sum() - 1) * 100 for s in SEASONS}
        cnt = (Cn[m][states].sum() / Cn["v3.1"][states].sum() - 1) * 100
        keep = m != "v3.1" and hi < 0 and all(v <= 0 for v in ps.values()) and cnt <= 1.0
        print(f"| {m} | {W[m].sum()/Wb.sum():.3f} | {(B.sum()/A0.sum()-1)*100:+.1f}% ({lo:+.1f} to {hi:+.1f}) | "
              + " | ".join(f"{ps[s]:+.1f}%" for s in SEASONS) + f" | {cnt:+.1f}% | {'KEEP' if keep else ('-' if m=='v3.1' else 'fails')} |")


if __name__ == "__main__":
    main()
