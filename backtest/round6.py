"""Idea 17 (backtest/V3_PROTOCOL.md): absolute humidity instead of the cold/dry weather index in the
regressions. Same primary score and rule as rounds 3-5 (log WIS, all 53 places, vs the CDC baseline).

    SENTINEL_WEATHER_FILE=weather_ah_state_week.csv python -m backtest.member_cache --tag memah
    SENTINEL_WEATHER_FILE=weather_ah_state_week.csv python -c "from backtest.exp_v3 import produce_all; \
        produce_all(tag='all53ah', families=('mem',))"
    python -m backtest.round6
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backtest.eval_all53 import HUB, load_family, wis_rows
from backtest.exp_v3 import SEASONS, quantile_average
from backtest.leaderboard import load_model
from backtest.member_cache import mean_members
from backtest.round3 import no_pr
from backtest.run_backtest import final_hosp

ROOT = Path(__file__).resolve().parents[1]


def main(alt_tag: str = "all53ah", label: str = "17. absolute humidity instead of cold/dry index"):
    mem, alt = load_family("mem"), load_family("mem", alt_tag)
    gbq, qar = load_family("gbq"), load_family("qar")
    refs = sorted(set(mem) & set(alt) & set(gbq) & set(qar))
    season = {r: mem[r]["season"] for r in refs}
    v3 = {r: quantile_average([mean_members(mem[r]["mem"]), gbq[r]["gbq"], no_pr(qar[r]["qar"])]) for r in refs}
    new = {r: quantile_average([mean_members(alt[r]["mem"]), gbq[r]["gbq"], no_pr(qar[r]["qar"])]) for r in refs}
    lin0 = {r: mean_members(mem[r]["mem"]) for r in refs}
    lin1 = {r: mean_members(alt[r]["mem"]) for r in refs}
    truth = final_hosp().set_index(["location", "date"])["value"]
    rows = []
    for r in refs:
        last = r - pd.Timedelta(weeks=1)
        for (loc, k) in v3[r]:
            y = truth.get((loc, last + pd.Timedelta(weeks=k)))
            if y is not None and not np.isnan(y) and (loc, k) in new[r]:
                rows.append((season[r], r, str(r.date()), loc, k - 1, float(y)))
    keys = pd.DataFrame(rows, columns=["season", "ref", "ref_s", "location", "horizon", "y"])
    idx = pd.MultiIndex.from_frame(keys[["ref_s", "location", "horizon"]])
    w = load_model(HUB / "model-output/FluSight-baseline", keys.rename(columns={"ref_s": "ref"})[["ref", "location", "horizon"]])
    w.index = w.index.set_names(["ref_s", "location", "horizon"])
    base = w.reindex(idx).values.astype(float)
    ok = ~np.isnan(base).any(axis=1)
    keys, base, y = keys[ok].reset_index(drop=True), base[ok], keys["y"].values[ok]
    lg = lambda a: np.log1p(np.maximum(a, 0))

    def sc(p):
        Q = np.round(np.vstack([p[r][(l, h + 1)] for r, l, h in zip(keys.ref, keys.location, keys.horizon)]))
        return wis_rows(lg(Q), lg(y)), wis_rows(Q, y)

    W0, C0 = sc(v3)
    W1, C1 = sc(new)
    L0, _ = sc(lin0)
    L1, _ = sc(lin1)
    Wb = wis_rows(lg(base), lg(y))
    wk = keys["ref"].values
    weeks = np.unique(wk)
    A = pd.Series(W0).groupby(wk).sum().reindex(weeks).values
    B = pd.Series(W1).groupby(wk).sum().reindex(weeks).values
    ii = np.random.default_rng(0).integers(0, len(weeks), size=(2000, len(weeks)))
    lo, hi = np.percentile((B[ii].sum(1) / A[ii].sum(1) - 1) * 100, [2.5, 97.5])
    ps = {s: (W1[(keys.season == s).values].sum() / W0[(keys.season == s).values].sum() - 1) * 100 for s in SEASONS}
    states = ~keys["location"].isin(["US", "11", "72"]).values
    cnt = (C1[states].sum() / C0[states].sum() - 1) * 100
    keep = hi < 0 and all(v <= 0 for v in ps.values()) and cnt <= 1.0
    lines = [f"# {label}", "",
             "| | score | change vs v3 (95% CI) | " + " | ".join(SEASONS) + " | counts, 50 states | rule |",
             "|---" * 7 + "|",
             f"| v3 | {W0.sum() / Wb.sum():.3f} | - | - | - | - | - | |",
             f"| {label} | {W1.sum() / Wb.sum():.3f} | {(B.sum() / A.sum() - 1) * 100:+.1f}% ({lo:+.1f} to {hi:+.1f}) | "
             + " | ".join(f"{ps[s]:+.1f}%" for s in SEASONS) + f" | {cnt:+.1f}% | {'KEEP' if keep else 'fails'} |", "",
             f"Regressions alone: {L0.sum() / Wb.sum():.3f} -> {L1.sum() / Wb.sum():.3f}"]
    out = "\n".join(lines)
    (ROOT / "backtest/round6.md").write_text(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
