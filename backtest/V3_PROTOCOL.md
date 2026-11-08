# Sentinel v3: the rules, written down BEFORE any v3 result was seen (October 7, 2026, 02:02 UTC)

## Why new rules are needed
v2 was designed on 2023-24 and 2024-25, then scored once on 2025-26 (0.789 vs the CDC baseline;
CDC ensemble 0.669). That one number is the official blind result of v2 and it stays in the
README whatever happens next. There is now **no unseen past season left**. The only fully fair
test from here on is the live 2026-27 season.

Anything changed from now on is tuned for the live season, so it may use all three past seasons
(that is what a real team does: learn from last winter). To keep the numbers honest anyway:

## The rules
1. **Ideas are listed here first**, before testing, and come from published methods of other
   teams or from how flu behaves, not from staring at where v2 failed in 2025-26. No idea is added
   to the list after results are seen without saying so in the results.
2. **Leave-one-season-out (LOSO).** Whenever an idea has a number to choose (a weight, a window),
   it is chosen on two seasons and scored on the third, for each of the three seasons in turn.
   The reported score of a season never uses that season to choose anything. This is the honest
   estimate of "how well does the *procedure* work on a season it hasn't seen".
3. **Keep an idea only if** its LOSO result is better than v2 overall with a 95% interval
   entirely below 0 (whole weeks resampled, 2000 times), AND it is not worse than v2 in any of
   the three seasons on its own. A gain that only shows up in one season is not kept.
4. **Everything is reported**, kept or not, with the same table format.
5. The leak test must pass on the final v3 design before it replaces v2 in the weekly run.
6. v3 numbers are labelled "leave-one-season-out", never "blind" or "held-out".

## The ideas (fixed list)
1. **Tree weight by horizon.** v2 blends the trees at 33% for every horizon. The design-season
   analysis already showed the trees help little at 0-1 weeks ahead and a lot at 3-4 weeks.
   Choose one weight per horizon (grid 0, 0.1, ..., 0.7) by LOSO.
2. **Median instead of mean when combining the 8 regressions.** The CDC's own ensemble uses
   the median of the teams' quantiles because it's robust to one member going wild.
3. **A third, different model family: direct quantile autoregression on rates**, following the
   method CMU published for its "TimeSeries" entry (hub file model-metadata/CMU-TimeSeries.yml):
   lagged values of the population-normalised series, put on a common ("whitened") scale so the
   10 seasons of FluSurv-NET and ILI history can be used as extra training rows, fitted jointly
   across states, using only training weeks within +/-3 weeks of the same time of year (all past
   years), one quantile regression per quantile level, sorted afterwards. v2's regressions model
   log-growth and get their ranges from past errors; this one models the level and fits each
   quantile directly, so its mistakes should differ. Blend weight chosen by LOSO.
4. **Two training windows** (2 and 3 years) averaged, for diversity. Mixed, not switched: the
   earlier "longer window" finding was hindsight and was reverted.
5. If compute allows: **more tree bags** (6 -> 20), which only reduces the trees' randomness.
6. If compute allows: **a "weeks from Christmas" clue for the tree model**, as UMass's flusion
   uses (`delta_xmas`), because holiday weeks distort reporting and care-seeking. Disclosure:
   this one is ALSO suggested by knowing that v2's worst 2025-26 weeks were just after New Year,
   so it is not purely "from outside". LOSO and the every-season rule apply as for the others.

(Ideas 3 and 6 were made concrete after reading the other teams' published method descriptions
in the hub's model-metadata folder, at about 02:05 UTC, still before any v3 result; the first
v3 result was produced at 02:23 UTC.)

## Added after the first results (about 02:26 UTC) -- disclosed, not pre-registered
First results: no idea passed rule 3. The leave-one-season-out runs showed WHY: the three seasons
disagree about which model family is best (the trees are much better than the regressions in
2025-26 and much worse in 2024-25), so a weight chosen on two seasons does badly on the third.
Two follow-ups, both decided before running them:
7. **Equal weights across model families** (regressions, trees, quantile autoregression), the
   standard "quantile average" used by flusion. No number to tune, so nothing can be overfitted
   to a season; motivated by the LOSO finding that estimated weights don't transfer. Kept if it
   beats v2 overall; reported with the caveat that the choice of families was made after seeing
   results. (The median of the three is reported too, but the mean is the pre-committed choice.)
8. **Weights that adapt during the season**: each week, every family is scored on its forecasts
   of the last 10 weeks that have already come true (against the data as published that week,
   never the final data), weight = half equal + half proportional to 1/its average WIS; equal
   weights until 3 weeks of results exist. Parameters fixed here, not tuned. Kept only if it beats
   idea 7 overall (95% interval below 0) and in every season.

## Results (October 7, 2026, about 02:35 UTC)
Change in total WIS vs v2 on the same 15,790 forecasts (50 states, 79 weeks, 3 seasons), 95%
interval from resampling whole weeks. "KEEP" would need the interval below 0 AND no season worse.
Full numbers: `backtest/v3_results.csv`, made by `python -m backtest.exp_v3`.

| Idea | Change vs v2 | 95% interval | 2023-24 | 2024-25 | 2025-26 | Rule 3 |
|---|---|---|---|---|---|---|
| 0. one tree weight, chosen LOSO | -0.4% | -4.6 to +5.0 | -6.3% | +7.7% | -4.6% | fails |
| 1. tree weight per horizon, LOSO | -2.3% | -7.2 to +3.7 | -6.2% | +6.9% | -8.5% | fails |
| 2. median of the 8 regressions | -0.1% | -0.3 to +0.1 | +0.1% | +0.2% | -0.4% | fails |
| 3. + quantile autoregression, weight LOSO | -1.0% | -4.6 to +3.2 | -0.2% | +3.6% | -5.4% | fails |
| 3b. same, weight per horizon | -2.0% | -5.6 to +2.1 | -1.2% | +2.8% | -6.6% | fails |
| 4. 2- and 3-year windows averaged | -0.0% | -0.9 to +0.7 | +0.0%* | -1.0% | +0.8% | fails |
| 7. **equal weights: regressions / trees / QAR** | **-6.5%** | **-13.5 to +0.5** | **-5.6%** | **+2.8%** | **-15.0%** | fails (narrowly) |
| 7b. same with the median | -7.5% | -19.1 to +3.6 | +1.4% | +5.0% | -22.6% | fails |
| 7c. equal weights: regressions / trees | -4.7% | -9.0 to -0.2 | -4.4% | +2.1% | -10.8% | fails (2024-25) |
| 8. adaptive weights, vs idea 7 | -0.2% | -0.4 to +0.0 | -0.2% | +0.0% | -0.4% | fails |

\* In 2023-24 there were not yet 3 years of data, so both windows were identical: idea 4 was in effect tested on two seasons.

Ideas 5 and 6 (more tree bags, a Christmas clue for the trees) were not run: each needs ~3-8 hours
of computing for the whole replay, and the analysis below suggests they can't change the decision.

What the numbers say: each model family alone, relative to Sentinel's own baseline:

| | 2023-24 | 2024-25 | 2025-26 |
|---|---|---|---|
| regressions alone | 0.731 | **0.496** | 0.929 |
| trees alone | **0.601** | 0.616 | **0.518** |
| quantile autoregression alone | 0.657 | 0.528 | 0.617 |
| v2 (67% regressions + 33% trees) | 0.624 | 0.492 | 0.749 |
| v3 (equal thirds) | 0.589 | 0.505 | 0.637 |

The seasons disagree about which family is best. In 2024-25 (the biggest season in 15 years, but a
regular shape) the regressions, which extrapolate growth, were best and the trees, which expect
seasons to turn like past ones, under-predicted the record peak. In 2025-26 (a record spike that
collapsed) the opposite: the regressions overshot by ~200% at the tip of the spike, the trees
called the turn. This is why weights chosen on two seasons fail on the third.

## Decision
**By the rules written above, no idea is kept.** Rule 3 asked for an interval entirely below 0
and no season worse; idea 7's interval reaches +0.5% and it is 2.8% worse in 2024-25.

**v3 = idea 7 anyway, as an openly declared judgment call**, for the live 2026-27 season:
- it has no weights to tune, so it cannot have been fitted to any season;
- worst case over three very different seasons: +2.8%; best case: -15%; total over the three seasons: -6.5%;
- on the two original design seasons combined it ties with v2 (-0.2%, interval -4.6% to +3.8%;
  2023-24 -5.6%, 2024-25 +2.8%), so it gives up nothing there in expectation, while in a season
  like 2025-26 it is much better (-15.0%, interval -23.8% to -1.1%). The case for v3 rests on
  that one season plus the principle below; that is exactly the kind of evidence rule 3 was
  written to distrust, which is why this is labelled a judgment call and not a test result.
  (All of this is on counts, 50 states. On the log scale the hub ranks with, all 53 places, the
  gain is smaller: v3 0.601 vs v2 0.609 over three seasons, -1.3%; 2023-24 0.619 vs 0.651,
  2024-25 0.594 vs 0.567, v3 worse; 2025-26 0.592 vs 0.610. See `backtest/all53.md`.);
- mixing model families that fail in different situations is the main lesson of the CDC's own
  ensemble and of every top team's design (flusion, CMU, Google all ensemble different families).

Honesty labels that follow from this:
- v2's 2025-26 score (0.789 vs the CDC baseline) remains Sentinel's only **blind** result.
- v3's numbers on all three seasons are **in-sample**: the choice of families and equal weights
  was made after seeing them. They are not evidence of how v3 will do in 2026-27.
- The live season is the test. A sealed v2 and v3 forecast made before the October 7 data release
  is in `submission/sealed/` so the comparison continues on data neither version has seen.

## Second independent review, and what followed (October 7, 2026, from about 03:30 UTC)
A second independent review re-checked v3. Every number in this file
reproduced. It found real problems, fixed as follows:
1. **Week-53 calendar bug** (all versions since the "already infected" clue): the season was computed
   from the calendar year, so a few January days that belong to ISO week 52/53 of the previous year
   (2016-01-02 in the history data, and live: 2027-01-02) were put in the wrong season, and the
   season-week count repeated a number after week 53. Live, this would have reset "admissions so
   far this season" right at the January 2027 peak. In the replay it changed only the trees and the
   QAR (through the 2015-16 history season's week numbers); the regressions were unaffected, because
   the hospital data start in February 2022 (0 of 120,432 regression forecasts changed). Fixed with one shared ISO-year calendar (`models/seasons.py`) in the regressions, the trees
   and the QAR. **Every replay cache was regenerated with the fix, and all results below were
   recomputed**; the earlier caches are kept in `backtest/cache/old_season_bug/` for comparison.
2. Trend probabilities: admissions are whole numbers, so a change of exactly 10 was counted as
   "stable"; boundaries now sit half-way between whole numbers.
3. The validator now also checks the hub's round rules (a real round, allowed target weeks, the
   Sunday-Wednesday submission window, US Eastern time); the weekly run skips weeks that are not
   rounds; the autopilot only submits this week's file.
4. Weather download restarts per place; archive values win over forecast-service values; the
   wastewater step warns about stale data.
5. The US, DC and Puerto Rico are submitted but were never scored. A replay of all 53 places is now
   part of the evaluation (`exp_v3.py --produce-all`).

### Ideas 9 and 10, written down before the regenerated results (about 03:45 UTC)
Disclosure: both come from the reviewer's partial look at 10 weeks of 2023-24 for the US, DC and
Puerto Rico (Puerto Rico: v3 1.43 and QAR alone 2.66 vs the CDC baseline; QAR's lowest quantiles
at 0 for the US today), so they are not blind.
9. **No QAR for Puerto Rico.** QAR learns "what happens at this time of year" pooled over mainland
   states and history; Puerto Rico's flu season follows a different (tropical) calendar, so QAR's
   central assumption doesn't hold there. For Puerto Rico only, v3 averages the regressions and the
   trees (the trees know each place). Kept if Puerto Rico's WIS over the three seasons improves.
10. **One spread factor for the final v3 ranges**, chosen leave-one-season-out from 0.80, 0.85, ...,
   1.10 (each quantile moved towards or away from the median). Rule 3 applies.

### Results after the fixes (about 07:20 UTC)
All caches were regenerated with the calendar fix. Effect of the fix on the earlier numbers: v2's
total WIS changed by +0.10% (2023-24), +0.07% (2024-25), +0.01% (2025-26); every v3 number above
is essentially unchanged (ideas 0-8 moved
by at most 0.2 points; the tables above show the regenerated numbers). v2's blind 2025-26 score
stays 0.789 vs the CDC baseline.

| Idea | Result | Rule | Kept? |
|---|---|---|---|
| 9. no QAR for Puerto Rico | Puerto Rico, log scale vs CDC baseline: 0.936 -> **0.854** (2023-24 1.013 -> 0.832, 2024-25 0.963 -> 0.877, 2025-26 0.804 -> 0.813); all 53 places, log: 0.602 -> 0.601 | PR's WIS over the three seasons improves | **yes** |
| 10. one spread factor for v3's ranges, LOSO | +0.9% (+0.0 to +1.7); worse in 2023-24 and 2025-26 | rule 3 | no |

v3 as submitted = idea 7 + idea 9. Puerto Rico remains Sentinel's weakest place (CDC ensemble 0.754
there).

### The all-53-places replay: the live setup, scored (`backtest/all53.md`)
Relative WIS vs the CDC baseline on identical forecasts (all 53 places the hub scores, 78 weeks: the
CDC published no baseline or ensemble for 2025-11-15):

| | log scale, 2025-26 | log scale, all 3 | counts, 2025-26 | counts, all 3 |
|---|---|---|---|---|
| v3 (in-sample) | 0.592 | 0.601 | 0.658 | 0.588 |
| v2 (2025-26 blind) | 0.610 | 0.609 | 0.781 | 0.631 |
| CDC ensemble | 0.616 | 0.650 | 0.655 | 0.662 |

Whole US (log scale, all 3 seasons): v3 0.436, v2 0.430, CDC ensemble 0.465. The QAR's wide US ranges
(the reviewer's point 3) cost little there: v3's 95% US range held the truth 99% of the time, too
wide, but its score is still better than the CDC ensemble's.

### Where v3 is weaker than v2 (descriptive; nothing was changed because of it)
- **By phase of the season** (v3 / v2 WIS, log scale, all 53 places, by month): v3 is clearly better
  around the peak and turn (Jan-Feb 2024 0.88 / 0.77, Jan 2026 0.85, Mar 2026 0.92) and weaker in the
  tails: late season April-May is worse in 5 of 6 months (up to +12%), early season is mixed
  (Oct 2023 +8.8%, Nov 2023 +4.7%, Nov 2024 +3.1%, Nov 2025 -8.4%). The trees and the QAR learned
  from ~10 older seasons that flu fades fast after the peak; in NHSN data it fades more slowly.
- **July-September 2026** (`backtest/recent_check.md`; corrected data for all versions; never used
  for any v3 decision): v3 was worse than v2 (50 states, relative to Sentinel's baseline: 0.674 vs
  0.645) and under-predicted the early rise of the US more (mean signed error of the 1-week-ahead
  median: v3 -10.3%, v2 -7.8%, v1 -10.6%). This season started unusually early and high, which
  level-based models expect less.
- So the first weeks of 2026-27 are in v3's weakest phase. Kept as is (a change chosen now would be
  tuned on these few months), but it is the first thing to re-check once 6-8 weeks of 2026-27 have
  been scored against the sealed and submitted forecasts.

## Round 3 ("v3.1"), written down before any of its results (October 7, 2026, 15:08 UTC)
Context: v3 goes out as is tonight (first live round, 2026-10-10). This round can only change what is
submitted from the second round on, and only if it passes.

**Primary score from now on: the CDC's ranking scale**, i.e. WIS on log(x+1), all 53 places, relative
to the CDC baseline (`backtest/eval_all53.py`). Counts on the 50 states are reported alongside.
**Rule:** keep an idea only if, on the primary score, the change vs v3 over the three seasons has a 95%
interval entirely below 0 (whole weeks resampled) AND it is not worse in any season; and it may not
make the 50-state count score worse by more than 1%.

11. **A fourth model family: pooled AR(6) on fourth-root rates**, the other half of UMass's flusion
    in 2024-25 and 2025-26 (hub file model-metadata/UMass-flusion.yml: "AR(6) pooled model with
    shared coefficients across locations and per-location variance", fit to NHSN only). Six lags of
    (rate per 100k + 0.01)^(1/4), one set of coefficients for all places, each place's own noise
    level, forecasts 1-4 weeks ahead by simulating 1,000 paths. Equal weights with the other three
    (a quarter each; Puerto Rico keeps its rule: no QAR).
12. **Hospital data count more in the trees and the QAR**: NHSN rows get weight 3 in training (history
    rows 1). Disclosure: motivated by the earlier finding that both fade too fast at the end of
    seasons, where the old FluSurv/ILI history declines faster than NHSN; the weight 3 is fixed here,
    not tuned. Tested for the QAR first (cheap); the trees only if compute allows.

### Round 3 results (about 15:20 UTC; `backtest/round3.md`, made by `python -m backtest.round3`)
Primary score, log WIS over all 53 places vs the CDC baseline; v3 = 0.601.

| Idea | Score | Change vs v3 (95% CI) | 2023-24 | 2024-25 | 2025-26 | 50 states, counts | Kept? |
|---|---|---|---|---|---|---|---|
| 11. + pooled AR(6), quarters | 0.614 | +2.2% (+0.4 to +4.0) | +1.3% | +0.4% | +4.6% | +4.1% | no |
| 12. QAR with hospital rows weighted 3 | 0.600 | -0.2% (-0.9 to +0.4) | +1.6% | -1.3% | -0.9% | +0.5% | no |
| both | 0.615 | +2.4% (+0.4 to +4.4) | +2.8% | -0.3% | +4.4% | +4.9% | no |

The AR(6) alone scores 0.794 (QAR alone 0.691, trees alone 0.633 over the same forecasts): this
simple version is far weaker than the Bayesian one UMass runs, and adding it only dilutes v3.
Weighting the hospital rows in the QAR helps 2024-25 and 2025-26 a little and hurts 2023-24 (the
season with only one earlier season of hospital data), so it fails the every-season rule. The trees
part of idea 12 was not run: the QAR result gives little reason to spend ~6 hours of computing on it.
**v3 stays unchanged.**

### Idea 13, written down before its result (about 15:23 UTC)
Descriptive look first (disclosed): on the primary score, v3 beats the CDC ensemble clearly when
the true count is above 30 (0.83-0.89 of the ensemble's WIS) but loses below 30 (1.02-1.05), and
those low counts are 37% of v3's total log score. There, v3's 50% ranges held the truth 68-71% of
the time: too wide.
13. **Narrower ranges for small forecasts**: when v3's median for a place and week is below 30
    admissions, every quantile's distance from the median is multiplied by a factor c chosen
    leave-one-season-out from {0.6, 0.7, 0.8, 0.9, 1.0} on the primary score; forecasts with a median
    of 30 or more are untouched. Round-3 rule applies (log, all 53 places).
**Idea 13 result** (15:35 UTC): 0.600, -0.2% (-0.5 to +0.2); 2023-24 +0.1%, 2024-25 -0.4%,
2025-26 -0.3%; factors chosen 0.8 / 0.9 / 0.9. Fails (interval crosses 0, 2023-24 worse). The ranges
for small counts were not the problem; v3 stays unchanged.

**Where this leaves v3:** 13 ideas tested since v2, 2 kept (equal weights, no QAR for Puerto Rico).
On the CDC's ranking scale v3 is at about the CDC ensemble's level or better in each season, and the
best teams of 2025-26 are about 4-5% better again. The remaining gap is at small counts (below 30
admissions, where v3 trails the CDC ensemble by 2-5%), and no cheap fix closed it. Larger steps
(a mechanistic or Bayesian hierarchical model, spatial features for the trees) are multi-day
projects; the next scheduled check is the live season itself after 6-8 weeks.

## Round 4, written down before its results (October 7, 2026, about 15:34 UTC)
Descriptive look first (disclosed): v3's median is too LOW more often than too high. Over all 53
places, the real number was above v3's median 60% of the time instead of 50%; by season
2023-24 50-55%, **2024-25 72-76%**, 2025-26 54-59%. Every family does it, more at longer horizons
(trees and QAR more than the regressions). The size of the bias changes from season to season, so a
fixed correction is unlikely to transfer, but a correction that **learns the bias during the season**
could.
14. **In-season bias correction of the whole v3 forecast.** Each week and for each horizon, take
    v3's forecasts of the last 6 weeks whose target week has already been published, compare their
    medians with the numbers AS PUBLISHED THAT WEEK (never final data), pooled over all places:
    b = median of log((published + 1) / (median + 1)). Multiply every quantile by exp(s * b), with
    s chosen leave-one-season-out from {0, 0.25, 0.5, 0.75, 1.0}. No correction until 3 weeks of
    results exist. Round-3 rule applies (log, all 53 places).
15. Reference only (not a candidate): the same with a fixed shift per horizon chosen
    leave-one-season-out, to show whether the in-season version adds anything.
**Round 4 results** (`backtest/round4.md`): both fail. 14 (in-season bias correction): +2.5% (+1.0
to +4.3), the leave-one-season-out choice was "no correction" for two seasons and 0.5 for 2025-26,
which made 2025-26 7.1% worse. 15 (fixed shift per horizon): +0.4% (-2.1 to +3.1), helps 2024-25
(-3.8%), hurts 2023-24 (+3.8%). The low bias is real but not predictable from recent weeks: it
comes from seasons running bigger or longer than expected, and correcting it backfires at turns.

### Idea 16, written down before its result (about 15:35 UTC)
16. **Last week's CDC ensemble as a fourth voice.** Every Thursday the CDC publishes its ensemble of
    all teams' forecasts (FluSight-ensemble). By the next Wednesday it is one week old, but its
    forecasts for 2, 3 and 4 weeks ahead cover this week's horizons 0, 1 and 2. Candidate: average it
    in as a fourth family with equal weight (a quarter) for horizons 0-2; horizon 3 unchanged.
    Nothing tuned. Disclosure: this would make Sentinel partly an ensemble of hub models, so the
    team card would have to say `ensemble_of_hub_models: true`, and the CDC may then leave Sentinel
    out of its own ensemble. Even if it passes, it is offered to the team as a choice, not switched
    on silently. Round-3 rule applies.
**Idea 16 result** (`backtest/round5.md`): 0.599, -0.4% (-1.7 to +1.1); 2023-24 -0.2%, 2024-25 -1.5%,
2025-26 +0.4%; counts +2.6%. Fails. Last week's CDC ensemble adds almost nothing to v3, so Sentinel
stays fully independent of other teams' forecasts.

### Idea 17, written down before its result (about 15:40 UTC)
17. **Absolute humidity instead of the cold/dry index.** The best-established weather driver of flu
    is absolute (specific) humidity: dry air in absolute terms (Shaman & Kohn, PNAS 2009; Shaman et
    al., PLoS Biology 2010), not temperature and relative humidity separately. It can be computed
    from the same daily temperature and relative humidity Sentinel already downloads:
    AH (g/m3) = 6.112 * exp(17.67 T / (T + 243.5)) * RH * 2.1674 / (273.15 + T).
    Weekly mean per state, standardised against the state's own past only (same past-only rule as
    now), sign flipped so higher = drier. It replaces the weather signal in the regressions (the
    other families don't use weather). Nothing tuned. Round-3 rule applies.
**Idea 17 result** (`backtest/round6.md`): 0.601, -0.1% (-0.3 to +0.1); 2023-24 +0.2%, 2024-25 +0.1%,
2025-26 -0.5%. Fails: absolute humidity carries the same information as the cold/dry index
(correlation 0.82 between the two weekly signals).

### Idea 18, written down before its result (about 15:44 UTC)
18. **Neighbouring states as clues for the trees**, after UMass's published 2025-26 change to flusion
    ("GBQR 3-source spatial2 model ... incorporating spatial correlation", hub model-metadata). For
    every series and week, two extra tree features: the average level (on the common scale) and the
    average 4-week slope of the bordering states in the same data source on the same date
    (`models/neighbors.py`; places without land neighbours get none). Neighbours didn't help the
    regressions (Experiment 1), but trees can use them differently (e.g. "my neighbours already
    turned"). Nothing tuned. Primary score and round-3 rule.

### Idea 6 (from the first list), made concrete before its result (about 15:45 UTC)
6. **"Weeks from Christmas" for the trees**: one extra tree feature, the number of weeks between a
   week and the week containing December 25 of its season (negative before, positive after), as
   flusion's `delta_xmas`. Holiday weeks change care-seeking and reporting. Disclosure as listed at
   the top (v2's worst weeks were just after New Year). Primary score and round-3 rule. Run on the
   all-53 replay alongside idea 18, each tested on its own against v3.

## Round 5, written down before its results (October 7, 2026, about 17:32 UTC)
First live evidence (sealed forecasts, week ending Oct 3; US actual 3,246): v2 median 3,124, v3 2,860,
baseline 2,515. Both beat the baseline; v2 beat v3, in line with v3's known early-season weakness.
Looking at why (descriptive): the newest week is first published ~7-9% too low (median, 2024-25 and
2025-26; mean 12-15%), and the regressions correct for that, **but the trees and the QAR are given the
uncorrected numbers** (`engine.forecast` passes them the data as published). They also show the larger
low bias (median 10% too low one week ahead, vs 6.5% for the regressions).
19. **Give the trees and the QAR the same corrected newest week as the regressions** (the existing
    reporting-delay fix, nothing new to tune). A consistency fix. Primary score and round-3 rule.
(Ideas 18 and 6, the tree variants, were interrupted by a machine restart at 17:16 UTC and are
re-run after this one.)

### Diagnosis of v3's under-forecast in the live rise, and ideas 20a-c (about 17:50 UTC, before their results)
Tonight's whole-US forecast (data through Oct 3, newest week 3,246 after 6 weeks of ~25% weekly
growth), one week ahead, by family: regressions 4,328; trees 3,681 (4,124 with the corrected newest
week); **QAR 3,264, i.e. no growth at all** (3,474 corrected). v3 = 3,757. With data through Sep 26 the
same pattern: QAR 2,509 vs an actual 3,246. The QAR learns "what happens at this time of year"
from ~15 past seasons, where a level this high in early October almost never happened and was
usually a blip that fell back, so it predicts a fall back: it can't recognise an early season. Over
the replay it also has the largest low bias of the three families (median 10-22% too low by horizon).
20a. **QAR trained on all in-season weeks** (season weeks 5-45, every year), not only +/-3 weeks
     around the same time of year, so it learns growth dynamics from far more situations,
     including early and fast seasons. One switch, nothing tuned.
20b. **QAR given the season's position explicitly instead of the window**: all in-season weeks, with
     the season week as an extra regressor (so seasonality is still there, but momentum is learned
     from every week). Nothing tuned.
20c. Reference: v3 without the QAR (regressions + trees, halves), to see what the QAR adds at all.
Primary score and round-3 rule; 20a/20b also combined with idea 19 once its trees finish.

### Results of ideas 19 and 20 (about 19:30 UTC; `python -m backtest.swap_eval`)
| Idea | Score (v3 0.601) | Change vs v3 (95% CI) | 2023-24 | 2024-25 | 2025-26 | 50 states, counts | Rule |
|---|---|---|---|---|---|---|---|
| 19. corrected newest week for trees AND QAR | 0.597 | -0.7% (-1.5 to +0.3) | -0.0% | -2.1% | +0.0% | +0.9% | fails (interval) |
| 19, trees part only | 0.598 | -0.6% (-1.0 to -0.1) | -0.1% | -1.3% | -0.4% | +0.2% | **passes** |
| 19, QAR part only | 0.599 | -0.3% (-0.7 to +0.2) | +0.0% | -1.1% | +0.1% | +0.5% | fails |
| 20a. QAR on all in-season weeks | 0.620 | +3.2% (+0.7 to +5.6) | +4.8% | -0.2% | +4.7% | +4.3% | fails |
| 20b. same + season-week term | 0.618 | +2.7% (+1.1 to +4.5) | +3.7% | +1.0% | +3.4% | +3.1% | fails |
| 20c. no QAR at all (reference) | 0.600 | -0.1% (-1.9 to +1.8) | +2.8% | -2.7% | -0.5% | +1.9% | fails |

**Decision: v3.1 = v3 with the trees given the corrected newest week.** Disclosure: idea 19 as written
(trees and QAR) fails by a hair; its two parts were computed side by side, and the trees part passes
the rule on its own (interval below 0, better in every season). The QAR keeps the published numbers.
In 2023-24 the archive has no revisions, so the fix can only matter in 2024-25 and 2025-26, which is
where it helps. Effect on tonight's whole-US forecast one week ahead: trees 3,681 -> 4,124, v3
3,757 -> about 3,905. QAR's time-of-year window is needed (20a/b are clearly worse), so its blindness
to early seasons stays a known weakness; without the QAR (20c) v3 is no better overall.
Before v3.1 replaces v3: end-to-end check against the replay and the leak test (rule 5).

### Checks of v3.1 and what's next (about 20:05 UTC)
- End-to-end: the engine run on replay weeks reproduces the replay bit for bit (400 of 400 for the
  50-state replay, 424 of 424 for the all-53 v3.1 replay) **when given the same input data**. With
  today's download the first try matched only half: the CDC revised scattered weeks of the ER-visit
  history (108 of 10,868 values, back to 2022) in today's release. Kept the older file in
  `data/history/old_downloads/` so the check can be reproduced (`SENTINEL_ED_FILE`).
- Leak test on v3.1: see `tests/results/leak_engine_v31.log`.
- Still queued, now tested on top of v3.1 (trees with corrected input), same rule: idea 12 for the
  trees (hospital rows weighted 3), idea 18 (neighbouring states), idea 6 (weeks from Christmas).

## Round 15 (2026-10-21): live overshoot fixes
Live weeks 1-2: our medians sat above truth in 56-71% of states (ensemble bias ~0); 70-100% of our gap
to the ensemble came from 8 small states (KY, NE, IN, MN, MT, WI, IA...). Replay shows the same
early-season small-state overshoot in 2023-24 and 2024-25 (median log error about +0.2) but the opposite
in 2025-26 (-0.18), so the sign is not stable.
- D damp positive growth in small places: +0.64% (CI +0.05..+1.24) fail
- E same, Oct-Nov only: +0.36% fail
- B constant shift on rising rows: +0.23% fail
- O online season-so-far bias correction: -0.08% (CI -1.59..+1.79), 2025-26 worse, fail
v3.1 unchanged.

## Round 16-17 (2026-10-21)
Diagnosis: in states under 100/week our median growth (+0.27 log) matched real growth (+0.22) on average;
the misses (NE, MN, WI, MT, IN) were states that dropped while we, like the ensemble, expected a rise.
That is mostly count noise in tiny states, not a fixable bias.
- L lagged FluSight-ensemble (last week's, horizon h+1) as a member: alone it is far worse (0.342 vs 0.253);
  blend -0.05% (CI -0.61..+0.50) fail
- S spread scaled by place size: -0.24% (CI -1.00..+0.47), 2025-26 worse, fail
v3.1 unchanged.

## Round 18 (2026-10-21): live data check
Live pipeline audit: on Oct 14 our ER file stopped at Oct 3 (the hub's Oct 10 ER week was published that
day), so the engine fell back to ER one week behind; wastewater raw is from Oct 4 (data.cdc.gov blocked
from the build machine), so it is carried forward flat. Rebuilding ref 2026-10-17 with the ER week that was
really available: h0 WIS 0.1714 vs 0.1707 submitted (+0.4%, 30 places better, 20 worse) -> no measurable
effect this week. Keep: fetch ER after the hub update so live matches the tested setup.

## Round 19 (2026-10-21): why top teams do better
Live decomposition (2 weeks, 48 teams scored on all three cases): in places >=100/week we are top-tier
(WIS 0.124; top teams 0.119-0.139). The whole gap is places <100/week: 0.192 vs ~0.17, median +0.10 log
high (top teams -0.07..+0.06), spread ~10% wider. Ensemble also overshot several of the same states.
Replay over full seasons shows the OPPOSITE in small places (all families biased low: regressions -0.18,
trees -0.08, QAR -0.18), so the live overshoot is early-season specific.
- W-size: family weights by place size, LOSO: -0.19% (CI -2.45..+2.16), small places -2.0% but 2023-24 and
  2024-25 worse -> fail
v3.1 unchanged.

## Round 20-21 (2026-10-21)
- 20 TEMP early-season shift (places <100, Oct-Nov, -0.05 log): replay affected rows -2.6% / -2.3% / +3.1%
  (2025-26 worse), whole season -0.14%; live ref 10-10 slightly worse (0.1789 -> 0.1798), ref 10-17 slightly
  better (0.1728 -> 0.1703). Coded in build_submission.py behind EARLY_SHIFT (expires 2026-11-30), not adopted.
- 21 family average on log scale instead of count scale: +2.39% worse (CI +1.27..+3.49), all seasons worse.
  It does cut the early small-place bias (+0.121 -> +0.083) but narrows ranges too much. fail

## Round 22 (2026-10-30), written down before its results: smoothing the starting point
Live clue (3 target weeks, states under 100/week): our median errors per state are NOT persistent from
one target week to the next; the same-horizon errors of consecutive weeks are negatively correlated
(h0: -0.08 and -0.33). That is the signature of a forecast anchored on a noisy newest week: a dip that
is pure count noise drags next week's forecast down, then the count bounces back. First a descriptive
check on the replay (lag-1 correlation of h0 errors, small vs big places), then:
  A1 shift every quantile (log(x+1) scale) by a * d, d = (local linear trend fitted to the last k
     as-published weeks, evaluated at the newest week) - (newest week); only places whose newest week
     is < T. a in {0,.25,.5,.75,1}, T in {50,100,200,1e9}, k in {3,4,5}, chosen leave-one-season-out.
  A2 same shift with a count-based weight instead of a cutoff: w = n0 / (n0 + newest), n0 in
     {5,10,20,40,80}, k in {3,4,5}, chosen leave-one-season-out.
Values as published that week (hub archive). Same keep rule as rounds 3-21.
**Round 22 results**: the replay does not show the live pattern: lag-1 correlation of h0 errors in places
under 100/week is +0.09 (2023-24 -0.08, 2024-25 +0.18, 2025-26 +0.08), so the live negative correlation is
3 weeks of noise, not a stable feature. A1 +0.58% (CI -0.43..+1.77), 2025-26 +2.4%: fail. A2 +0.76%
(CI -0.06..+1.63): fail. v3.1 unchanged.

## Round 23 (2026-10-30), written down before its results: one side of the range at a time
Live (targets Oct 10 and 17, scored on the data as revised by Oct 28, which is fairer to us than the
first-published numbers): our medians are unbiased in places under 50/week; the gap is the WIDTH of our
95% range in places under 100/week: 2.62 / 2.30 / 2.07 (log units, buckets <20, 20-50, 50-100) vs
2.24 / 1.85 / 1.70 for the CDC ensemble. In places >= 100/week we are near the best (0.087 vs 0.082-0.096).
Earlier symmetric narrowing failed because surges need the upper tail (round 8 C, round 11, round 17).
Untested so far: narrowing the two sides separately.
  T1 scale the lower half (quantiles below the median) by fl and the upper half by fu, per horizon,
     on the log(x+1) scale around the median; fl, fu in {.7,.8,.9,1,1.1}, chosen leave-one-season-out.
  T2 the same, only for places whose as-published newest week is < 100; others unchanged.
Also descriptive: width, coverage and WIS of each family in small places (which family is widest).
Same keep rule as rounds 3-22.
**Round 23 results**: lower side of the range is too long almost everywhere (truth below our 2.5% line 1.4%
of the time; QAR 0.2%). T1 -1.02% (CI -2.41..+0.49), 2025-26 +0.16%: fail. T2 (small places only) -1.35%
(CI -2.18..-0.48), but 2023-24 +0.27%: fail by the every-season rule. The chosen lower factor was 0.7, the
edge of the grid, in all 12 folds. By phase (replay, small places): Oct-Nov too high (truth above our 75%
line 9.5% of the time), Dec-May too low with the lower tail too long. Live 2026-27 (revised data): every
team's small-state ranges are too wide (no truth outside anyone's 95% range), ours the widest.

## Round 24 (2026-10-30), written down before its results: fix the widest family, not the ensemble
In the replay the QAR's 95% range covers 97.8% in small places (99.7% in Oct-Nov), width 3.78 vs 2.4-2.5
for the other two families; its lower tail is almost never reached (0.2%).
  T2b T2 again with the grid extended because 0.7 was at its edge in every fold (disclosed second look):
      fl in {.4,.5,.6,.7,.8,.9,1}, fu in {.8,.9,1,1.1,1.2}.
  Q1  QAR's quantiles pulled toward its own median by fq (log scale) before the family average,
      fq in {.4,.5,.6,.7,.8,.9,1}, per horizon, leave-one-season-out.
  Q2  same with separate factors for QAR's lower and upper halves, each in {.4,.6,.8,1}.
  Q3  QAR keeps its median but takes the average spread (log-scale deviations from the median) of the
      regressions and the trees. Nothing tuned.
Same keep rule as rounds 3-23.
**Round 24 results** (`python -m backtest.round24`):
| Idea | Change vs v3.1 (95% CI) | 2023-24 | 2024-25 | 2025-26 | small places | Rule |
|---|---|---|---|---|---|---|
| T2b (grid extended, second look) | -0.54% (-2.02 to +1.12) | +2.96% | -2.59% | -1.85% | -0.86% | fail |
| Q1 QAR both sides, one factor | +0.58% (-0.15 to +1.27) | -0.42% | +0.35% | +1.66% | +0.82% | fail |
| **Q2 QAR lower / upper separately** | **-1.77% (-2.65 to -0.79)** | **-1.89%** | **-3.38%** | **-0.29%** | -1.83% | **passes** |
| Q3 QAR median + other families' spread | +0.03% (-0.66 to +0.71) | -1.49% | +0.52% | +0.95% | +0.34% | fail |
Q2 chose QAR lower factor 0.4 and upper 1.0 in 10 of 12 folds (0.8 upper in 2 folds); on all three seasons the
best choice is (0.4, 1.0) at every horizon. Fixed (0.4, 1.0): -1.91% (-1.89 / -3.81 / -0.29% by season); 50-state
counts -0.28% (2025-26 +2.27%, total within the 1% limit); 95% coverage 0.963 -> 0.946; truth below the 2.5% line
1.3% -> 3.0%. Second look below the grid (disclosed, not used): 0.3 -> -2.14%, 0.2 -> -2.28% but 2025-26 +0.01%,
so the pre-registered 0.4 stays. Live check (this season's submitted forecasts, QAR recomputed from the data as
published each week, scored on data as of Oct 28): -2.03% (ref 10-10 -2.37%, 10-17 -2.62%, 10-24 +0.06%; places
<100/week -1.76%, >=100 -4.74%). **Decision: v3.2 = v3.1 + QAR lower half x0.4 (engine.V32, FINAL from the
2026-11-07 round).** The transform in models/engine.py was checked against the tested one (identical).

## Round 25 (2026-10-30), written down before its results: the same for the other two families
Replay, small places: the trees' upper tail is too short (truth above their 97.5% line 7.5% of the time; big places
8.7%), the regressions' 5.7%. On top of v3.2:
  R1 trees: lower half x fl in {.6,.8,1}, upper half x fu in {1,1.2,1.4,1.6}, per horizon, leave-one-season-out
  R2 regressions: same grids
  R3 R1 and R2 together (each family's factors chosen leave-one-season-out jointly per horizon)
Same keep rule, compared with v3.2.
**Round 25 results**: R1 trees +0.31% (CI -0.40..+1.09), R2 regressions +1.00% (CI +0.03..+2.11), R3 both +1.64%
(CI +0.21..+3.17): all fail; the chosen factors jump between folds, unlike the QAR's (0.4 in 10 of 12 folds).
v3.2 unchanged.
**Round 12-13 results, recovered 2026-10-30** (re-run; their printouts were lost): round 12's analog query used
final data for the last 5 weeks (a leak, fixed in round 13). Round 12: AN alone +18.4%, V4 -3.1% (CI -5.2..-0.9,
+0.3% in 2023-24 and 2025-26), V4h -1.0%: fail. Round 13 (query as published): B1 -0.5% (CI -2.0..+1.1; 2025-26
+3.0%), B2 -0.7% (CI -1.9..+0.6; 2025-26 +1.9%): fail. Round 10 (re-run): M (MultiQT) +2.2%, K (climatology) +0.7%,
X (Christmas) +0.1%: fail. Round 14 (re-run): S1-S3 regional pooling chose a = 0 in every fold: no gain.

## Round 26 (2026-10-30), written down before its results: an ER-visit anchor for small states
The ER share of flu visits rests on far more visits than the admissions count, so in small states it is a much
less noisy reading of where the epidemic stands. Two of the three families (trees, QAR) never see it.
ER-implied newest week = newest ER share x median(admissions / ER share) over the 4 weeks before the newest.
d = log(1 + implied) - log(1 + newest as published). On top of v3.2:
  E1 shift all quantiles (log(x+1) scale) by a * d for places with newest < T; a in {0,.25,.5,.75,1},
     T in {50,100,200}, chosen leave-one-season-out
  E2 weight w = n0 / (n0 + newest) instead of a cutoff, n0 in {5,10,20,40,80}, leave-one-season-out
ER values: data/clean/ed_flu.csv (final; the hub archive only has ER versions from Dec 2025, and ER revisions
are rare: 108 of 10,868 values changed in the Oct 2026 release). Same keep rule, compared with v3.2.
**Round 26 first results, NOT valid**: E1 -5.12% (CI -6.83..-3.38), every season better, a = 0.5 / T = 200 in every
fold; E2 -1.29% (fail). Leak found before any use: the newest ER week is revised up later (first published vs final,
1,717 place-weeks from Dec 2025: median +2.6%, mean +4.4% on the log scale), and the test used final ER values.
Re-run (26b) with ER as first published: the hub archive's ER versions where they exist (refs from 2025-12-20),
otherwise final ER divided by exp(that place's median first-to-final revision, pooled median if < 5 cases)
for the newest week only (older weeks are essentially final). Nothing else changes.
**Round 26b results** (ER as first published): E1 -4.81% (CI -6.45..-3.15), 2023-24 -4.36%, 2024-25 -7.46%,
2025-26 -3.00%, a = 0.5 / T = 200 in every fold; rows with archived ER only (2025-26 from Dec) -3.11%: passes the
rule. E2 -1.30% (CI -3.49..+1.11): fail. **But the live check fails**: on this season's forecasts (refs 10-10, 10-17,
10-24; ER and hospital data exactly as published those Wednesdays; v3.2 as the base; scored on data as of Oct 28)
E1 is +5.95% worse (+3.5%, +13.8%, -1.3% by week; places < 100/week +7.8%, 100-200 -13.2%). The replay agrees that
the gain is in Dec-May (-4% to -10% for places < 200) and small in Oct-Nov (-0.3% to -1.5%; 100-200: +11.3%, n=58),
and grows with the ER flu share (newest week < 0.5%: -2.8%, 2025-26 +1.2%; 4%+: -15%). Not adopted as is.
## Round 26c, written down before its results (disclosed: motivated by the live check and the breakdown above)
The ER anchor only makes sense when the ER share rests on enough flu visits. E3 = E1 (a, T) applied only when the
place's newest ER flu share (as published) >= g; g in {0, 0.5%, 1%, 2%}, (a, T) as E1, all chosen leave-one-season-
out. Then the same live check. Adopt only if both the rule and the live check hold.
**Round 26c results**: leave-one-season-out picks no gate (g = 0) in every fold, i.e. E1 again (-4.81%, passes the
rule, fails the live check). Fixed gates, a = 0.5, T = 200:
| gate (newest ER flu share) | replay, change vs v3.2 (95% CI) | 2023-24 | 2024-25 | 2025-26 | 50-state counts | live, change vs v3.2 | rows shifted live |
|---|---|---|---|---|---|---|---|
| none | -4.81% (-6.45 to -3.15) | -4.36% | -7.46% | -3.00% | | +5.95% | 90% |
| 0.5% | -4.09% | -3.63% | -5.53% | -3.31% | | +1.73% | 31% |
| **1%** | **-3.10% (-4.33 to -1.87)** | **-3.46%** | **-3.79%** | **-2.21%** | **-1.30% (every season better)** | **+0.10%** | 10% |
| 2% | -2.32% (-3.37 to -1.33) | -2.18% | -2.74% | -2.10% | -1.06% | -0.27% | 2% |
**Decision: v3.3 = v3.2 + ER anchor with a 1% gate (engine.V33 = FINAL; applied in submission/build_submission.py).**
Disclosure: the gate was chosen after seeing the live check. It gives up about a third of the replay gain to stay
out of the regime where the live check failed (right now the newest ER flu share is 0.3-0.5% in the median small
state, so the anchor is mostly off and switches on as flu picks up). 95% coverage 0.946 -> 0.951. The engine
function reproduces the tested d exactly (203 of 203 rows with archived ER). Monitoring: after each scored week,
compare the anchored places with and without the anchor; switch it off if it is worse over the first 4 scored
weeks in which it is active.

## Round 27 (2026-10-30), written down before its results: the ER anchor in bigger places
T = 200 was the edge of round 26's grid in every fold. On top of v3.3's gate (1%):
  E4 (a, T) with a in {.25,.5,.75}, T in {200, 400, 800, no limit}, leave-one-season-out. Same rule, vs v3.3.
**Round 27 results**: E4 -0.26% (CI -1.85..+1.24), 2025-26 +0.75%: fail. T stays 200. v3.3 unchanged.
**Caveat on round 26 (found after the decision, logged as is):** before Dec 2025 the CDC published ER visits on
Fridays, two days after the deadline, so for 2023-24, 2024-25 and early 2025-26 the anchor test simulates TODAY's
schedule (ER published Wednesday with the hospital data), using final ER scaled down by its typical first-to-final
revision. That is the right question for the live forecast, but those seasons are a simulation of the input. On the
23 weeks with real archived Wednesday ER (2025-26 from Dec 20): gate 1% -2.44% (CI -5.78..+0.54), no gate -3.11%
(CI -7.43..+1.11): same direction, not significant on its own. The monitoring rule above is what protects us.

## Round 28 (2026-10-30), written down before its results: neighbouring states for the trees, on top of v3.3
Idea 18 (round 7) re-run on the corrected-input trees (`--produce-all gbqspbf`, restarted twice after stalls).
N1 = v3.3 with the trees swapped for the neighbour-feature trees; everything else identical (QAR tail, ER anchor).
Same keep rule, compared with v3.3.
**End-to-end check of v3.3 (2026-10-30):** `python -m submission.build_submission --reference-date 2026-10-31`
with today's data reproduces the submitted 2026-10-31 v3.1 file exactly where v3.3 should change nothing (all 164
non-anchored medians, Puerto Rico, the US), raises the lower quantiles (QAR fix; 2.5% quantile +0.39 on the log
scale on average) and moves the 12 places the ER anchor applies to (e.g. Alaska h0 median 11 -> 18: its newest week
fell from 22 to 7 admissions while its ER flu share stayed at 1.6%; Nevada 74 -> 95: ER flu share 0.93% -> 1.36%
in one week with flat admissions). Format check VALID. The build also writes the no-anchor shadow file for
`python -m backtest.score_er_anchor`.
**Round 28 results** (all 79 weeks): N1 -0.87% (CI -1.24..-0.52), 2023-24 -1.10%, 2024-25 -0.77%, 2025-26 -0.76%;
50-state counts -0.69% (every season better): passes. **Decision: v3.4 = v3.3 + neighbour-feature trees
(engine.V34 = FINAL).** No live check is possible for this one without rebuilding the trees on each week's data as
published (about an hour per week); the end-to-end build below checks the code path.
**Total, v3.4 vs v3.1 (replay, all 53 places, log WIS):** -5.78% (CI -7.50..-3.97); 2023-24 -6.33%, 2024-25 -8.17%,
2025-26 -3.23%; places < 100/week -6.74%, others -4.15%. Live so far (Oct 10-24, before the ER anchor switches on):
about -2% (the QAR fix only).
**End-to-end check of v3.4 (2026-10-31):** rebuild of ref 2026-10-31 runs and validates (VALID); vs v3.3 the medians
move by 6% typically (largest 29%), US medians 7,971 / 9,687 / 11,790 / 13,927 (v3.3: 7,897 / 9,895 / 12,217 / 14,460).
The test build's shadow file was deleted (the real 10-31 submission was v3.1, so it must not enter the ER-anchor check).

## Round 29 (2026-10-31), written down before its results: ideas from the literature review
Source: literature review (Rumack, Tibshirani & Rosenfeld 2022 recalibration; Lichtendahl et al. 2013 and the
hubEnsembles FluSight case study on quantile averaging vs linear pools; Jose et al. 2014 exterior trimming).
Base = v3.4. Phase of each forecast, from data known at forecast time: 1-week log change of the place's ER flu
share as published (hospital admissions as published if no ER): rising > +0.1, falling < -0.1, else flat.
  P1  Beta recalibration of the final forecast per (phase x horizon): PIT of the truth on the log(x+1) scale from
      the 23 quantiles (linear between them), Beta(a, b) fitted by maximum likelihood on the other two seasons,
      new quantile at level t = old quantile function at G^-1(t) (linear extrapolation outside 1%-99%,
      G^-1 limited to [0.001, 0.999]). Primary. P1h: the same moved half-way (level 0.5 G^-1(t) + 0.5 t).
  P2  Mix of quantile averaging and linear pool (average of the families' CDFs on the log(x+1) scale), weight
      lambda in {0,.25,.5,.75,1} per phase, chosen leave-one-season-out; ER anchor applied after, as in v3.4.
  P4  Exterior trimming in places < 100/week: below the median, mean of the two highest family quantiles;
      above, mean of the two lowest; median = mean of all three. P4u: upper side only. Nothing tuned.
Same keep rule (95% CI below 0, no season worse, 50-state counts not worse by > 1%), compared with v3.4.
**Round 29 results**: P1 beta recalibration by phase x horizon +3.99% (CI +1.86..+6.11), P1h +1.33%, P2 linear-pool mix
+0.45% (CI +0.13..+0.80), P4 exterior trimming +0.50%, P4u +1.19%: all fail. v3.4 unchanged.
**Vintage timing audit (2026-10-31, after the forums review flagged a known trap):** in data/raw/time-series.csv
`as_of` is the data week's Saturday for all 60 versions before 2025-07-23 (as_of = newest week) and the Wednesday
publication date for all 41 versions after (newest week + 4 days). The replay picks versions by their newest week,
not by the label, so the label change does not shift data in time. Whether those older versions were public by the
Wednesday deadline: CDC's FluSight page "as of March 19, 2025" (a Wednesday) shows data through March 15, i.e. the
newest week was public by the deadline in 2024-25, as the replay assumes. Not verifiable from the archive alone:
whether each pre-July-2025 version is the Wednesday release or a later re-capture of the same week.

## Round 30 (2026-10-31), written down before its results: corrected newest week for the QAR, inside v3.4
Round 5 (idea 19, QAR part) failed by a hair on v3 (-0.3%, CI -0.7..+0.2). Two reviews cite UMass's 2025-26 nowcast
experiment (AR6 0.855 -> 0.820 with a corrected newest week). The base is now different (QAR lower tail shrunk,
neighbour trees, ER anchor), so it is re-tested once, disclosed as a second look:
  C1  v3.4 with the QAR fed the backfill-corrected newest week (cached `all53qarbf`), same tail factor (0.4, 1.0).
Same keep rule, compared with v3.4.
**Round 30 results**: C1 +0.21% (CI -0.30..+0.77), 2025-26 +1.33%, 50-state counts +1.01%: fail. v3.4 unchanged.

## Round 31 (2026-10-31), written down before its results: national and season-position clues for the trees
From the top-team review (DMAPRIME-HYB: national level and change as inputs; Schnitzels-stack5: time of season,
distance from the season's peak so far, level relative to the US; GT_Influencers: position against the season's
peak so far; UMass-peakGB). Added to v3.4's trees (`models/gbq.py`, `seasonpos` + `national`), all from data up to
each week only: running season maximum of the series, distance below it, weeks since it; average level and 4-week
slope of all places of the same source that week, and the place's level relative to that average.
  N2  v3.4 with the trees swapped for these (`--produce-all gbqspnbf`). Same keep rule, compared with v3.4.
One combined variant (one overnight replay), disclosed.
**Round 31 results**: N2 -1.68% (CI -2.68..-0.63), per season -2.63% / -2.13% / -0.51%, small places -2.50%,
50-state counts +0.19% (inside the 1% limit): **keep -> v3.5** (`engine.V35`, flag `gbq_context`).
By month (all seasons): Oct -7.1%, Nov -2.9%, Dec -3.0%, Jan +2.7%, Feb -4.9%, Mar -0.3%, Apr -1.6%, May -0.9%.
Peak month (January) slightly worse; disclosed. In past Oct-Nov weeks the new trees moved medians down a little
(median log shift -0.034). Test build for 2026-10-31 (not submitted; that week was v3.1): VALID, same 12 anchored
places, US medians 8,136 / 10,322 / 12,981 / 16,026 (v3.4: 7,971 / 9,687 / 11,790 / 13,927); across places the
median rises 3% / 5% / 7% / 10% by horizon, the largest move DC (+33% at h1, 6 -> 8). The engine uses the same
`gbq_forecast` call as the replay, with `seasonpos` and `national` on.

## Round 32 (2026-11-01), written down before its results: ER visits inside the trees
From the top-team review: Delphi (CMU) reports its largest single gain from adding NSSP ER visits to its models; the
UMass and Georgia Tech write-ups also feed ER visits into their tree models. Today Sentinel uses ER visits only in the
regressions and in the ER anchor. Added to v3.5's trees (`models/gbq.py`, `ed=True`), hospital rows only (history
rows have no ER data and stay missing): the place's log ER flu share relative to its own in-season mean, its 3- and
5-week slopes, its 1-week change, and the 2-week change of (log ER share - log admissions rate). ER data as published
each Wednesday: CDC's archived version where one exists (Dec 2025 on); before that, today's schedule is simulated
(final numbers up to the newest hospital week, the newest week divided by the typical later revision, 1.028), as in
round 26c, disclosed.
  E1  v3.5 with the trees swapped for these (`--produce-all gbqedbf`). Same keep rule, compared with v3.5.
  Secondary, not used for the decision: the same without the ER anchor (does the anchor still add anything?).
One variant (one overnight replay).
**Round 32 results**: E1 -0.68% (CI -1.15..-0.20), per season -0.31% / -1.54% / -0.29%, small places -0.57%,
50-state counts -1.04%: **keep -> v3.6** (`engine.V36`, flag `gbq_ed`). By month: Oct -2.8%, Nov -1.5%, Dec +0.6%,
Jan -1.6%, Feb -0.4%, Mar -0.1%, Apr -0.6%, May -0.7%. Disclosed: on the 23 weeks with CDC's archived ER versions
(from 2025-12-17) the gain is -0.35% (CI -1.44..+0.62); on the weeks with the simulated Wednesday schedule, -0.83%.
Secondary: with ER trees, the ER anchor still helps (-2.68%, CI -3.95..-1.44), so it stays.
Engine safeguard: if the ER file has no newest week on submission day, the trees run without ER (v3.5 trees) and the
build prints a warning. Test build for 2026-10-31 (not submitted): VALID, US medians 8,057 / 10,324 / 12,936 / 15,811
(v3.5: 8,136 / 10,322 / 12,981 / 16,026); median change across places -1% to -3% by horizon, range -19% to +9%.

## Round 33 (2026-11-02), written down before its results: ER visits inside the QAR
Delphi's CMU-TimeSeries (the published method our QAR follows) gained most from adding NSSP ER visits as a lagged
covariate. Added to the QAR (`models/qar.py`, `ed=True`): the 3-week slope of the log ER flu share and the 2-week
change of (log ER share - log admissions rate), both as in round 32, plus a has-ER indicator. Rows without ER data
(history sources, hospital weeks before Oct 2022, gaps) get 0 and indicator 0, so the ER coefficients are learned from
hospital rows only. Same ER data as round 32 (archived versions, else today's schedule simulated).
  Q1  v3.6 with the QAR swapped for this one, same lower-tail factor (0.4, 1.0) (`--produce-all qared`).
Same keep rule, compared with v3.6. One variant.
Amended before any score was seen: in the smoke test the has-ER indicator acted as an intercept for the only season
with ER data in the early replay weeks (2022-23, which fell in December), cutting the US 1-week median for 2023-12-09
from 8,012 to 5,507. The indicator is removed; the two ER terms stay (0 where missing; both are changes, centred near
0, so the zero rows do not pull their coefficients).
**Round 33 results**: Q1 -1.72% (CI -3.71..+0.27), per season +1.56% / -6.68% / -0.53%, small places -1.17%,
50-state counts -2.16%: **fail** (CI includes 0, 2023-24 worse). v3.6 unchanged. By month: Oct +3.4%, Nov +1.0%,
Dec -7.4%, Jan -1.8%, Feb -2.1%, Mar -0.8%, Apr +1.3%, May -3.1%. Note, not a decision: in 2023-24 only one past
season had ER data to learn the coefficients from; today there are four. Not re-tested (no new evidence possible
until this season's data is in); a candidate to look at again after the 2026-27 season.
