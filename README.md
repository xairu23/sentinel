# Sentinel

Weekly probabilistic forecasts of influenza hospital admissions for all 50 US states, DC, Puerto Rico and the US, built for the CDC's public [FluSight forecasting hub](https://github.com/cdcepi/FluSight-forecast-hub). COVID-19 and RSV come next.

**Research question:** does adding wastewater surveillance data improve state-level influenza hospitalization forecasts?

## Results: real-time replay of 3 past seasons

Every past week was forecast with the hospital data **exactly as it was published that week** (the CDC hub archives every weekly version), and scored against final data with the CDC's rule, the weighted interval score (WIS). Relative WIS below 1 = better than the CDC baseline.

**Design discipline, and where it ends:**
- **v2** was designed on 2023-24 and 2024-25 only; 2025-26 was held out and scored once. **v2 on 2025-26 is Sentinel's only blind result.**
- **v3** (what is submitted now) was chosen after that, using all three seasons, under rules written down in advance ([`backtest/V3_PROTOCOL.md`](backtest/V3_PROTOCOL.md)). Its numbers are **in-sample**: they show what v3 is, not how it will do next season.

| Model | All 3 seasons | 2023-24 | 2024-25 | 2025-26 |
|---|---|---|---|---|
| **Sentinel v3** (submitted; in-sample) | **0.596** | **0.614** | 0.533 | 0.669 |
| Sentinel v2 (2025-26 blind)¹ | 0.638 | 0.651 | **0.519** | 0.789 |
| CDC FluSight ensemble (official, live) | 0.677 | 0.740 | 0.657 | 0.669 |
| Sentinel v1 (first design) | 0.721 | 0.760 | 0.546 | 0.933 |
| CDC FluSight baseline | 1.000 | 1.000 | 1.000 | 1.000 |

Compared on 15,590 identical (week, state, horizon) forecasts ([`backtest/vs_cdc_v3.md`](backtest/vs_cdc_v3.md)). ¹ Re-run with the week-53 calendar fix found by the second review (below); as first scored: 0.637 / 0.650 / 0.518 / 0.789.

**Read this table honestly:**
- On the **one blind test** (v2, 2025-26), Sentinel was **behind the CDC ensemble** on this count scale (0.789 vs 0.669), though level with it on the log scale the CDC uses for rankings (below).
- v3 ties the CDC ensemble on 2025-26, but v3 was chosen after seeing that season.
- The "all 3 seasons" column is mostly seasons Sentinel was designed on. The live 2026-27 season is the real test; sealed v2 and v3 forecasts made before the October 7 data release are in [`submission/sealed/`](submission/sealed/).

**v2 vs v1 on identical forecasts** (95% confidence interval, bootstrap over forecast weeks):
- held-out 2025-26: **−15.5% (−18.9% to −9.7%)**
- design seasons: −8.5% (−13.0% to −4.7%)

The improvement on the unseen season is at least as large as on the design seasons, which is the opposite of what overfitting looks like.

**First independent review (v2).** A separate review of the code, done fresh without the design notes, reproduced the held-out result exactly and found three things, all now fixed:
1. **ER-visit data was assumed published the same week in 2023-25, but back then it came out two days after the forecast deadline.** The replay now uses same-week ER only where CDC's archive shows it was published that way (from the 2025-26 season); otherwise it is one week behind. This cost the design seasons about 2-3% and left the held-out season unchanged.
2. **A gap-filling order bug let wastewater be used a week early in 8 of ~4,000 forecasts.** Fixed in both v1 and v2.
3. **Gaps in the leak test**, which now uses CDC's archived hospital data and also checks that the newest wastewater week is never used.

**Second independent review (v3).** Another fresh reviewer re-checked v3: every number reproduced, and it found one real bug. Flu seasons were computed from the calendar year, so some January Saturdays that belong to the previous ISO year's week 52 or 53 landed in the wrong season. Live, that would have reset "admissions so far this season" at the January 2027 peak. Fixed with one shared ISO calendar ([`models/seasons.py`](models/seasons.py)), and every replay was re-run (scores moved by at most 0.1%). It also pointed out that the US, DC and Puerto Rico had never been scored (now they are, below), plus smaller fixes to the trend target, the hub-rule checks and the downloads ([`backtest/V3_PROTOCOL.md`](backtest/V3_PROTOCOL.md)).

Also worth knowing:
- Two v1 features shared by v2 (peak-dampening and the bias correction) were chosen while looking at 2025-26, so v2's absolute 2025-26 score is somewhat optimistic. The v2-vs-v1 difference is not affected.
- The tree model's design follows UMass's flusion, a top team in these same seasons.

**Rank among all hub teams, the CDC's way** ([`backtest/leaderboard_official.md`](backtest/leaderboard_official.md)): pairwise relative WIS (the method of Cramer et al. 2022 used in FluSight evaluations), models covering at least 75% of Sentinel's forecasts, on two scales. Scope: the 50 states and Sentinel's 79 replay weeks, not the CDC's full official set. On **counts**, big states and big weeks dominate; on **log counts**, the scale the hub says final rankings use, every state counts about equally and errors are relative.

| Season | v2, counts | v2, log (CDC ranking scale) | v3, counts (in-sample) | v3, log (in-sample) | CDC ensemble, counts / log |
|---|---|---|---|---|---|
| 2023-24 | 2 of 28 (0.648) | 2 of 28 (0.645) | 2 of 28 (0.611) | 1 of 28 (0.612) | 0.738 / 0.718 |
| 2024-25 | 1 of 39 (0.518) | 1 of 39 (0.560) | 1 of 39 (0.532) | 1 of 39 (0.588) | 0.654 / 0.621 |
| 2025-26 | 23 of 43 (0.791), **blind** | **8 of 43 (0.610), blind** | 14 of 43 (0.671) | 6 of 43 (0.594) | 0.667 / 0.616 |

So on the one blind season, v2 was behind the CDC ensemble on counts (count scores are dominated by the biggest states in the biggest weeks, which is where v2's January overshoot happened) but level with it on the log scale (0.610 vs 0.616), 8th of 43. Best on the log scale that season: Google_SAI-FluEns (0.569). (The simpler ratio-of-means leaderboard, [`backtest/leaderboard_v3.md`](backtest/leaderboard_v3.md), gives the same count-scale ranks.)

**All 53 places, the exact live setup** ([`backtest/all53.md`](backtest/all53.md)). The tables above cover the 50 states. The hub also scores the US, DC and Puerto Rico, and the live forecast trains on all 53 places, so the whole replay was run that way too (relative WIS vs the CDC baseline, identical forecasts):

| | log, 2025-26 | log, all 3 | counts, 2025-26 | counts, all 3 |
|---|---|---|---|---|
| Sentinel v3 (in-sample) | 0.592 | 0.601 | 0.658 | 0.588 |
| Sentinel v2 (2025-26 blind) | 0.610 | 0.609 | 0.781 | 0.631 |
| CDC ensemble | 0.616 | 0.650 | 0.655 | 0.662 |

Whole US, log scale, all 3 seasons: v3 0.436, v2 0.430, CDC ensemble 0.465. Weakest place: Puerto Rico (v3 0.854 vs the CDC ensemble's 0.754), whose flu season doesn't follow the mainland calendar.

**Where v2 failed, and why v3 exists:** in 2025-26, two forecast weeks (3 and 10 January 2026) made 40% of v2's error for the whole season. US admissions jumped +78% in one week (21,057 → 37,505) and then stopped almost dead (42,505, then 29,912). v2's regressions, which extrapolate growth, overshot by ~200% at the tip of the spike; the tree model called the turn. In 2024-25 it was the other way round: the regressions were best and the trees under-predicted the record peak. Each model family fails in a different kind of season, so v3 gives three families an equal say instead of a tuned mix (tuned mixes chosen on two seasons failed on the third every time; see [`backtest/V3_PROTOCOL.md`](backtest/V3_PROTOCOL.md)). How much v3 gains over v2 depends on the scale: on counts (50 states) −6.5% over three seasons; on the log scale the CDC ranks with (all 53 places) only −1.3% (0.601 vs 0.609), and in 2024-25 v3 is worse there (0.594 vs 0.567).

**First live result** (forecasts sealed before the data existed; [`backtest/live_scoreboard.md`](backtest/live_scoreboard.md), updated every week): week ending October 3, 2026, whole US: actual 3,246; v2 3,124, v3 2,860, flat baseline 2,515. Both versions beat the baseline easily; v2 beat v3, which is what led to the v3.1 fix below.

**Ideas tested after v3 that did not pass** (all on the CDC's ranking scale, rules written before each result; details in [`backtest/V3_PROTOCOL.md`](backtest/V3_PROTOCOL.md)): a fourth model (simple AR(6), +2.2% worse), hospital data weighted more in the QAR (-0.2%, noise), narrower ranges for small counts (-0.2%, noise), learning the low bias during the season (+2.5% worse), a fixed upward shift (+0.4%), last week's CDC ensemble as a fourth voice (-0.4%, noise), absolute humidity instead of the cold/dry index (-0.1%), the QAR trained on all weeks instead of the same time of year (+3.2% / +2.7% worse), no QAR at all (-0.1%).

**Where v3 is weaker:** it beats v2 around the peak and turn of a season but loses in the tails (late season worse in 5 of 6 months; early season mixed), and in July-September 2026 it under-predicted the early rise more than v2 (US, 1 week ahead: median about 10% too low on average vs 8% for v2; 50 states: 0.674 vs 0.645 relative to Sentinel's baseline; [`backtest/recent_check.md`](backtest/recent_check.md)). So the first weeks of 2026-27 are v3's weakest phase; it's the first thing to re-check once the live season has 6-8 scored weeks.

**Caveats:**
- These are replays, not live submissions. The live 2026-27 season is the real test.
- The replay covers 79 of 89 weeks: CDC's archive has no version for 10 of them (2024-25: Nov 30, Dec 14, Jan 11, Jan 25; 2025-26: Oct 11 to Nov 8 and Dec 6), so the start of 2025-26 is not in any score.
- For 2023-24 and 2024-25, CDC's archive labels each version by its newest week, not its release date, so the replay assumes the version ending with last week is what teams had on Wednesday. From 2025-26 on, the archive records the real release dates.
- The 50-state replay trains on the 50 states; the live forecast trains on all 53 places (the US, DC and Puerto Rico too). The all-53 replay above runs exactly the live setup.
- Wastewater has no archived versions, so it is used with a one-week delay.
- ER visits are used for the same week only where CDC's archive shows they were published that way; otherwise they are one week behind, with the newest week scaled to its typical first-publication value.

## Charts

The latest forecast (redrawn by every weekly run; right now the sealed forecast of October 7):

![Latest Sentinel forecast](figures/latest_forecast.png)

The replay of three past seasons, forecasting the week right after the latest data each time:

![Replay of past seasons](figures/backtest_replay.png)

## What Sentinel v3 does

`models/engine.py` is the single engine used by both the replay and the weekly forecast, so the code that is tested is the code that submits (checked by [`backtest/check_engine_vs_replay.py`](backtest/check_engine_vs_replay.py): run end-to-end on replay weeks, the engine gives bit-identical forecasts to the replay, 400 of 400 for the 50-state replay and 424 of 424 for the all-53-places replay of v3.1, given the same input data; the CDC sometimes revises old ER-visit numbers, so the check pins the older download).

1. **Reporting-delay fix:** the newest published week is usually revised up later. The engine corrects it by how much recent weeks were revised, learned from versions published before the forecast date only.
2. **Signals with their real publication delays:**
   - ER-visit share for the same week whenever it's published that way, as CDC's hub has done since the 2025-26 season; otherwise one week behind (detected automatically);
   - wastewater one week behind;
   - weather.
3. **Eight pooled log-growth regressions** (every combination of ER / wastewater / weather). Features:
   - recent growth (3 lags), time of year, level vs. usual, peak-dampening terms;
   - **cumulative admissions this season per 100k** ("already infected", the reason epidemics turn over).

   Recent weeks count more (half-life 52 weeks).
4. **Intervals** from past errors, widened when a state has been jumpy, sized half-way to each location's own error level, and nudged by 30% of recent bias.
5. **A second model family, gradient-boosted quantile trees** (`models/gbq.py`, after UMass's flusion):
   - trained on the hospital data plus ~10 seasons (2010–2020) of CDC FluSurv-NET hospitalisation rates and state ILI+, all on a common scale;
   - bagged over seasons, 6 bags.
6. **A third model family, quantile autoregression** (`models/qar.py`, after the method CMU published for CMU-TimeSeries): the last 3 weeks' levels on the same common scale (so the history counts too), trained only on the same time of year (±3 weeks, all past years), each of the 23 quantiles fitted directly.
7. **v3.1 (October 7 evening): the trees get the corrected newest week too.** The tree model was fed the newest week as first published, which is typically 7-9% too low and gets revised up; the regressions already corrected for that. Fixing it gave 0.6% on the CDC's ranking scale over three seasons (95% interval -1.0% to -0.1%, better in every season; [`backtest/V3_PROTOCOL.md`](backtest/V3_PROTOCOL.md), idea 19). Tonight's whole-US forecast one week ahead moved from 3,757 to about 3,905.
8. **Final forecast = equal-weight average of the three families** (quantile by quantile), except in Puerto Rico, whose flu season doesn't follow the mainland calendar that the quantile autoregression learns: there it's the regressions and the trees only. v2 used 67% regressions + 33% trees; v3's equal weights have nothing to tune (why: [`backtest/V3_PROTOCOL.md`](backtest/V3_PROTOCOL.md)).

Every forecast value is a whole number (a hub rule) and is checked against the hub's plausibility bound (no more than 30% of a place's population) before anything is sent.

The file also carries the hub's optional **trend target** (`wk flu hosp rate change`: probabilities of large decrease / decrease / stable / increase / large increase), read off the same 23 quantiles ([`submission/rate_change.py`](submission/rate_change.py)). Its category rules reproduce the CDC's own categories for every past week (32,767 of 32,767), and in the replay its Brier score was better than the CDC ensemble's trend forecasts (0.42 vs 0.51 in 2024-25, 0.42 vs 0.46 in 2025-26; in-sample, since v3 was chosen after seeing these seasons, though the trend step itself has nothing tuned; [`backtest/rate_change_eval.py`](backtest/rate_change_eval.py)).

## Experiments (decided on 2023-24 + 2024-25 only)

| Change | Effect (95% CI) | Kept? |
|---|---|---|
| Weather (cold/dry), on top of ER + wastewater | −5.6% (−9.4% to −1.4%) | yes |
| ER visits for the same week (vs. one week late) | −2.9% (−5.5% to +0.4%)* | yes, where it's really published that way |
| Recent weeks count more (half-life 52 weeks) | −1.9% (−3.8% to +0.4%) | yes, as part of the combination |
| "Already infected this season" features | −0.3% (−0.65% to −0.05%) | yes |
| Reporting-delay fix | −0.2% (−0.8% to +0.6%) | yes (more correct data) |
| **All four together** | **−4.5% (−7.6% to −1.6%)*** | **yes** |
| ~10 seasons of history added to the regressions | −0.6% to +1.3% (6 variants) | no: a straight-line model can't use it |
| **History-trained tree model, 33% of the blend** | **−5.9% (−11.9% to −0.6%)** | **yes** |
| Per-location interval width | −0.5% (−1.0% to −0.1%) | yes |
| National growth as a feature | +9.2% | no |
| Weighting big states more in the fit | +1.4% to +3.2% | no |
| Wider intervals | +2% to +6% | no (already calibrated) |
| Neighbouring states / lab-test positivity | +1.0% / −0.4%, unclear | no |
| Trees trained on Sentinel's 3 seasons only | +22.5% | no |

**v3 experiments** (all three seasons, leave-one-season-out; change vs v2; [`backtest/V3_PROTOCOL.md`](backtest/V3_PROTOCOL.md)):

| Idea | Change vs v2 (95% CI) | 2023-24 / 2024-25 / 2025-26 | Kept? |
|---|---|---|---|
| Tree weight chosen on 2 seasons, per horizon | −2.3% (−7.2 to +3.7) | −6.2 / +6.9 / −8.5% | no |
| Quantile autoregression added, weight chosen on 2 seasons | −1.0% (−4.6 to +3.2) | −0.2 / +3.6 / −5.4% | no |
| One spread factor for v3's ranges, chosen on 2 seasons (vs v3, not v2) | +0.9% (+0.0 to +1.7) | +1.3 / +0.0 / +1.7% | no |
| No QAR for Puerto Rico (QAR assumes mainland seasons) | Puerto Rico: 0.936 → 0.854 vs the CDC baseline, log scale | | yes |
| Median instead of mean of the 8 regressions | −0.1% (−0.3 to +0.1) | | no |
| 2- and 3-year training windows averaged | −0.0% (−0.9 to +0.7) | | no |
| Weights adapting within the season (vs equal weights) | −0.2% (−0.4 to +0.0) | | no |
| **Equal weights: regressions / trees / quantile autoregression** | **−6.5% (−13.5 to +0.5)** | −5.6 / +2.8 / −15.0% | **yes, as a declared judgment call** (it narrowly fails the pre-written rule) |

\* Measured assuming same-week ER data in 2023-25. The independent review showed it wasn't published in time back then, so those two gains are overstated for the design seasons; the replay now uses same-week ER only where the archive shows it existed. Live, from the 2025-26 season on, CDC's hub publishes ER visits for the same week, so the engine uses them.

## Wastewater finding (preliminary)

Comparing models that differ only by the wastewater signal: hospital trends → + wastewater −1.1% (−5.7% to +4.5%); hospital trends + ER visits → + wastewater −1.0% (−4.2% to +2.5%). **No clear effect.** Wastewater tracks hospital admissions closely (median correlation 0.78) but adds little once recent hospital trends are known. A first version that accidentally used future wastewater information showed a 15% gain; that leak was found and removed.

## Leak tests

- `python -m tests.test_no_future_leak` (v1 models)
- `python -m tests.test_engine_no_future_leak` (the engine as submitted: reporting-delay fix, all three model families, the history; for the 50 states and for all 53 places)

Both check that every forecast is identical when everything published after the forecast week is deleted. The engine test also checks that the newest wastewater week is never used and that ER visits fall back to one week behind when they weren't published in time. Logs: [`tests/results/`](tests/results/).

## How it works

```
scripts/ingest.py               download hospital data, ER visits and CDC's archive of every published
                                version (hub), wastewater (data.cdc.gov), weather (Open-Meteo)
scripts/wastewater_clean.py     one weekly wastewater number per state, using each site's past only
scripts/weather_clean.py        one weekly cold/dry-ness number per state, using each state's past only
scripts/build_history.py        2010-2022 FluSurv-NET + ILI+ history (data/history/, from UMass flusion, MIT)
models/engine.py                the forecasting engine (replay AND weekly forecast); V2 and V3 configs
models/gbq.py                   the history-trained gradient-boosted quantile trees
models/qar.py                   the quantile autoregression (v3's third model family)
models/models.py                v1 models and shared feature code
backtest/harness.py             fast real-time replay used for all experiments
backtest/experiments.py, exp2.py, exp_gbq.py   the v2 experiments above
backtest/run_v2.py              official v2 replay (baseline, v1, v2 on identical forecasts)
backtest/member_cache.py        every regression's forecast for every replay week (for v3 experiments)
backtest/exp_v3.py              v3 experiments (leave-one-season-out) and the official v3 results file
backtest/V3_PROTOCOL.md         the v3 rules, written before the results, and the decision
backtest/eval_all53.py          all 53 places (incl. US, DC, Puerto Rico) in the exact live setup
backtest/check_engine_vs_replay.py  the engine end-to-end == the replay, bit for bit
backtest/compare_cdc.py         score the CDC's real past forecasts the same way and compare
backtest/leaderboard.py         rank Sentinel's replay against every team in the hub
submission/build_submission.py  this week's file in the exact hub format (runs the engine)
submission/validate.py          safety checks before anything is sent (hub format, whole numbers, plausibility)
submission/sealed/              forecasts made before the truth was published, to check later
scripts/weekly_run.py           the whole weekly job in one command
.github/workflows/weekly_forecast.yml  the autopilot (every Wednesday, submission switched off by default)
```

A plain-language walkthrough of every part is in [`HOW_IT_WORKS.md`](HOW_IT_WORKS.md).

## Run it

```bash
pip install -r requirements.txt
python scripts/weekly_run.py                                   # data, forecast, validation, charts
git clone --depth 1 https://github.com/cdcepi/FluSight-forecast-hub ../FluSight-forecast-hub
python -m backtest.run_v2                                      # fair replay: baseline, v1, v2 (slow: trees)
python -m backtest.member_cache                                # the 8 regressions, every replay week
python -m backtest.exp_gbq --seasons 2023-24 2024-25 2025-26 --tag gbq6 --bags 6   # trees (hours)
python -m backtest.exp_v3 --produce-qar                        # quantile autoregression (~15 min)
python -m backtest.exp_v3                                      # all v3 experiments (seconds)
python -m backtest.exp_v3 --export backtest/results_v3.csv     # baseline, v1, v2, v3 on identical forecasts
python -m backtest.compare_cdc --results backtest/results_v3.csv --out backtest/vs_cdc_v3.md
python -m backtest.leaderboard --results backtest/results_v3.csv --out backtest/leaderboard_v3.md
```

## Data sources
- Hospital admissions, ER visits, archive of published versions: [CDC FluSight hub target data](https://github.com/cdcepi/FluSight-forecast-hub/tree/main/target-data)
- Wastewater: [CDC Wastewater Data for Influenza A](https://data.cdc.gov/Public-Health-Surveillance/CDC-Wastewater-Data-for-Influenza-A/ymmh-divb/about_data)
- Weather: [Open-Meteo](https://open-meteo.com/) historical and forecast APIs (history kept in `data/history/weather_daily.csv`, only recent weeks downloaded each run)
- History (2010–2022): CDC FluSurv-NET, ILINet and WHO/NREVSS clinical labs, via the [UMass Reich Lab flusion repository](https://github.com/reichlab/flusion) (MIT licence)

## Status
- [x] Data ingestion (hospital, ER visits, wastewater, weather, published-version archive)
- [x] v2 engine, real-time replay of 3 seasons with a held-out season, comparison with the CDC, leaderboard
- [x] v3: third model family, equal-weight ensemble, pre-written rules, sealed forecasts
- [x] Submission builder + validator, weekly autopilot (off until switched on)
- [ ] Register the team in the hub, first live submission
- [x] Charts on this page (redrawn by every weekly run)
- [ ] COVID-19 and RSV, preprint
