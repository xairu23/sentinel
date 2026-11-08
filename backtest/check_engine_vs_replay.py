"""Is the replay really what the engine does? Runs models/engine.py end-to-end (engine.FINAL) on a few
replay weeks and compares every forecast with the combination of the cached family forecasts that the
replay results are computed from, for the 50-state replay and for the all-53-places replay.

    python -m backtest.check_engine_vs_replay
"""
import numpy as np
import pandas as pd

from backtest.eval_all53 import load_family
from backtest.exp_v3 import quantile_average
from backtest.harness import Replay
from backtest.member_cache import load, mean_members
from backtest.run_v2 import EdArchive, make_v2_forecaster
from models import engine
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEEKS = ("2024-12-28", "2026-01-10")


def main():
    rep = Replay(("2024-25", "2025-26"))
    picks = [w for w in rep.weeks if str(w.ref.date()) in WEEKS]
    cfg = engine.FINAL
    skip = lambda q: {k: v for k, v in q.items() if k[0] not in cfg.qar_skip}
    # 1. the 50-state replay (trees on the data as published, i.e. v3; v3.1's corrected-input trees were
    #    replayed for all 53 places only)
    mem, gbq, qar = load("mem"), load("gbq6"), load("qar")
    f = make_v2_forecaster(rep, engine.V3)
    tot = same = 0
    for w in picks:
        a = f(w)
        b = quantile_average([mean_members(mem[w.ref]["members"]), gbq[w.ref]["gbq"], skip(qar[w.ref]["qar"])])
        tot += len(b)
        same += sum(np.array_equal(a[k], b[k]) for k in b)
    print(f"50-state replay: {same} of {tot} forecasts bit-identical", flush=True)
    # 2. the all-53 replay (exactly the live setup)
    tree_tag = "gbqbf" if cfg.trees_corrected else "gbq"
    m53, g53, q53 = load_family("mem"), load_family(tree_tag), load_family("qar")
    ed = EdArchive(ROOT.parent / "FluSight-forecast-hub", rep.sig["ed"])
    lags = engine.lag_table(rep.vint.t)
    pop = pd.read_csv(ROOT / "data/clean/locations.csv", dtype={"location": str}).set_index("location")["population"]
    tot = same = 0
    for w in picks:
        sig = {"ww": rep.sig["ww"], "weather": rep.sig["weather"], "ed": ed.as_published(w.as_of, w.last)}
        a = engine.forecast(w.v, w.last, w.as_of, sig, lags, pop, cfg, states_only=False)
        b = quantile_average([mean_members(m53[w.ref]["mem"]), g53[w.ref][tree_tag], skip(q53[w.ref]["qar"])])
        tot += len(b)
        same += sum(np.array_equal(a[k], b[k]) for k in b)
    print(f"all-53 replay ({'v3.1' if cfg.trees_corrected else 'v3'}): {same} of {tot} forecasts bit-identical")


if __name__ == "__main__":
    main()
