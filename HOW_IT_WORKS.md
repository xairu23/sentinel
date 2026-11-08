# How Sentinel works, explained simply

Sentinel is a program that guesses **how many people in each US state will go to hospital with flu** in the next 4 weeks, sends those guesses to the CDC every week, and gets graded on what really happens.

Think of it as a **weather forecast for flu**: instead of "70% chance of rain," it says "next week Texas will probably have about 300 flu patients in hospital, almost surely between 90 and 650."

---

## The big picture: 6 machines in a row

```
1. Shopping robot      → downloads the numbers (hospital, ER visits, sewage, weather,
                         and the CDC's archive of every past version of the hospital numbers)
2. Cleaners            → turn 300,000 sewage samples and daily weather into 1 number per state per week
3. The guessers        → make the predictions (Sentinel v3: 3 different kinds of models, averaged)
4. The practice room   → replays past winters to see how good the guessers are
5. The packing machine → puts this week's guesses in the CDC's exact format and checks it
6. The autopilot       → does 1, 2, 3 and 5 by itself every Wednesday
```

---

## Machine 1: the shopping robot (`scripts/ingest.py`)

**Its job:** go to the internet and bring back the numbers.

It visits 6 places:

1. **The list of places** (`locations.csv`): the 53 places we forecast (50 states, Washington DC, Puerto Rico and the whole US). Each has a **code number** called a FIPS code: California is `06`, Texas is `48`, the whole country is `US`. The CDC uses these codes instead of names, so we do too.
2. **The hospital numbers** (`hosp_flu.csv`): for every place and every week since February 2022, how many people went to hospital because of flu. This is the number we have to guess. It comes straight from the CDC's own forecasting project, so we are graded on exactly the same numbers as everyone else.
3. **The ER-visit numbers** (`ed_flu.csv`): every week, what share of all emergency-room visits in each state were for flu (for example 0.0027 = 0.27%). People often go to the ER **before** they get sick enough to be admitted to hospital, so this can be an early warning. It also comes from the CDC's forecasting project.
4. **The sewage numbers**: every sample of sewage tested for flu virus in US treatment plants. It comes from the CDC's website in "pages" of 50,000 rows at a time (like reading a long book chapter by chapter), with a short pause between pages so we don't overload their website.
5. **The archive of published versions** (`time-series.csv`): every weekly version of the hospital numbers exactly as the CDC first published it. Since v2, Sentinel uses it to learn how much the newest week usually gets revised up later.
6. **The weather** (Open-Meteo, free): daily temperature and humidity at one point per state. The full history since 2021 is kept in the repository (`data/history/weather_daily.csv`), and each week only the last few weeks are downloaded and added. (Downloading 5 years for 51 places every week would hit Open-Meteo's free limits.) If the weather website is down, the forecast still runs with the stored weather.

**Small details:**
- Each week is named by its **last day, a Saturday**. "Week 2026-09-26" means the week from Sunday 20 to Saturday 26 September. The CDC does the same.
- It saves a "raw" copy (exactly as downloaded) and a "clean" copy (sorted, right dates, right codes).

---

## Machine 2: the sewage cleaner (`scripts/wastewater_clean.py`)

**The problem:** there are about 300,000 sewage samples from hundreds of treatment plants. But every plant measures differently: different machines, different units (some per liter of water, some per gram of sludge). You can't compare plant A's number with plant B's, like comparing a temperature in °C with one in °F.

**The trick: compare each plant only with itself.**

Step by step:
1. For each sample, take the amount of flu virus found. If the lab found nothing, the CDC already puts a small "minimum" value instead of zero.
2. Use the **logarithm** of that amount. Virus amounts jump from 100 to 1,000,000, and the logarithm squishes this into a calm scale (2 to 6), where "10 times more" always looks like the same step.
3. Ask: **"Is today's amount high or low compared with this plant's own past?"** This gives a **z-score**:
   - 0 means normal for this plant,
   - +2 means much higher than usual,
   - −1 means a bit lower than usual.
4. **Only the plant's past is used, never its future.** (See "The cheating bug" below: this rule matters a lot.) A plant needs at least 20 past samples before we trust its score.
5. Very extreme scores are **capped at +4 / −4**, so one strange sample from a new plant can't break everything.
6. Put each sample into its **week** (ending Saturday) and average the plant's samples in that week.
7. Make **one number per state per week**: the average of all its plants, where **bigger cities count more** (a plant serving 1,000,000 people matters more than one serving 7,000).

**Result:** 9,054 weekly numbers covering 50 states, from November 2021 to October 2026. Sewage moves together with hospital admissions (a typical state matches at 0.78 out of 1), and in 16 states it moves about one week earlier (`scripts/lead_check.py`, chart `figures/wastewater_vs_hosp.png`).

---

## Machine 3: the guessers (`models/models.py`)

There are three guessers. Each gives not one number but **23 numbers**, called quantiles: from "very low" (1% chance it's below this) to "very high" (99% chance it's below this). The middle one (50%) is the best single guess.

Why 23 numbers? Because the CDC wants to know **how sure** you are. "300, maybe 280 to 320" is a very confident guess; "300, maybe 50 to 900" is a very unsure one.

### Guesser 1: the lazy baseline
- "Next week will be the same as this week."
- For the range, it looks at how much this state's numbers usually jump from one week to the next, and makes the range wider the further ahead it guesses.
- It's not smart. It's the **bar to beat**: if you can't beat it, your model is useless.

### Guesser 2: the growth guesser (no sewage)
It looks at a few clues and learns from past winters how those clues predict the future. The clues:

| Clue | In simple words |
|---|---|
| `d1` | How much the numbers went up or down **last week** |
| `d2` | How much they moved **the week before** |
| `d3` | How much they moved **3 weeks ago** |
| `sin`, `cos` | **What time of year it is.** Flu rises in autumn, peaks in winter, falls in spring. These two numbers are a clever way to tell the computer "where we are on the calendar circle" |
| `rel` | **How high the state is compared with its normal level** over the last 2 years. If it's already very high, it usually can't keep climbing forever |

**How it learns:** it uses a method called **linear regression** (OLS). Imagine drawing the best straight line through a cloud of points: the computer finds how much each clue matters. For example, "if last week went up a lot and it's December, next week usually goes up too."

**Important details:**
- It learns from **all states together** ("pooled"), so a small state learns from big states' patterns.
- It learns only from the **last 3 years** and is retrained **every week**.
- It predicts the **change** (on the log scale), not the number itself. Going from 100 to 200 and from 1,000 to 2,000 are both "doubling", the same kind of change.
- It's trained separately for 1, 2, 3 and 4 weeks ahead.
- **For the range:** it looks at how wrong it was in the past (its "errors") and uses those errors to make the range. If it was often wrong by ±30%, the range is about ±30%.
- A safety cap stops any guess from predicting more than about **7 times more or less** in the coming weeks, because that's not realistic.

### Guessers 3, 4 and 5: the same, plus extra signals
They are **exactly Guesser 2** with extra clues added:

| Guesser | Extra clues |
|---|---|
| `growth_ww` | **sewage**: how much it changed last week (`ww1`), the week before (`ww2`), and how high it is now (`wwl`) |
| `growth_ed` | **ER visits**: how much the share changed last week (`ed1`), the week before (`ed2`), and how high it is now (`edl`, on the log scale) |
| `growth_ed_ww` | **both** ER visits and sewage |

**Both extra signals are used one week late on purpose.** In real life, labs take time to report sewage results, and the old versions of the ER data were not saved before December 2025. So we pretend we always get them one week late. That can only make Sentinel look *worse*, never better: no cheating possible.

Why so many guessers? Because the research question needs **pairs that differ by only one thing**: growth vs growth_ww, and growth_ed vs growth_ed_ww. The only difference is sewage, so any difference in score is caused by the sewage.

### The final guesser: the Sentinel ensemble
Average the 23 numbers of **all four growth guessers**. **A team of guessers usually beats each one alone**, because their mistakes partly cancel out. This rule was fixed in advance (not picked after seeing which one scored best), and the average is what gets sent to the CDC.

For the whole US, Washington DC and Puerto Rico, which have no sewage number, the ensemble averages only the guessers that don't need sewage.

---

## Sentinel v2: what changed (October 2026, `models/engine.py` + `models/gbq.py`)

Everything above still describes the core of the regressions. **v2** added the changes below; the version that runs now is **v3** (next section), which is v2 plus a third model family. Every change below was tested on the 2023-24 and 2024-25 seasons only, kept only if the gain was statistically real, and only then checked on 2025-26 (see "Experiment log" and "The final exam").

**One engine for everything.** `models/engine.py` is the single piece of code used by the practice room AND the weekly forecast, so what gets tested is exactly what gets sent. (Checked: on the same weeks, the engine and the experiment code give identical numbers, 0 differences out of 600 forecasts.)

What the engine does, in order:

1. **Fix the newest week (reporting delay).** The CDC's first number for the newest week is usually too low: hospitals keep reporting late, so it gets revised **up** later (typically +5%, sometimes much more). The engine looks at how much the newest week was revised in the last 10 weeks (using only versions published before today) and corrects it.
2. **Use the ER numbers for the same week, when they really exist.** Since the 2025-26 season, the CDC's hub publishes ER visits on the same day and for the same newest week as the hospital numbers, so v1's one-week delay was throwing away their head start. Before that (2023-25), the ER numbers came out on Fridays, two days after the Wednesday deadline, so they really were one week behind. The engine checks automatically: if the newest ER week is the same as the newest hospital week, use it; if not, use ER one week behind. In the practice room, weeks without the CDC's archived ER version get the one-week-behind version, with its newest week scaled down by its typical later revision (about 2.8%).
3. **Two new clues: "how many have already been infected this season".** `cum` = total flu admissions so far this season per 100,000 people. As more people have had flu, fewer are left to catch it, which is the real reason an epidemic peaks and turns. `cum_d1` lets "fast growth + already many infected" mean "turning soon".
4. **Recent weeks count more.** Training still uses 3 years, but a week from a year ago counts half as much as this week (half-life 52 weeks), so the model adapts faster to how *this* season behaves.
5. **Ranges sized per place.** v1 used one shared error size for every place, so the whole-US forecast (smooth, big numbers) got ranges as wide as a tiny noisy state's. Now each place's range is moved halfway towards its own past error size.
6. **A second, very different guesser: the tree model (`models/gbq.py`).** This is the biggest change. It copies the idea of UMass's "flusion" model, one of the best teams in the CDC hub:
   - It learns from **~10 extra flu seasons** (2010-2020): CDC FluSurv-NET hospitalisation rates from 14 sites and "ILI+" (flu-like-illness visits x share of flu tests positive) for 49 states. That's hundreds of past peaks and turns, versus 3 seasons for the regressions. (History data: public CDC files kept in UMass's repository, MIT licence, see `data/history/`.)
   - Every series is put on a common scale (4th root, then divided by its own typical high level, then centred) so a Utah outpatient curve and a California hospital curve can teach each other about *shape*.
   - Instead of a straight line, it uses **gradient-boosted decision trees** (LightGBM): many small yes/no rules like "if growth has slowed for 2 weeks and the level is near its usual peak, expect a drop". Trees can learn shapes a straight line can't.
   - One tree model per quantile level, each trained 6 times on different random 70% of seasons ("bagging"), and the middle of the 6 answers is kept.
   - **Final answer = 67% regressions + 33% tree model.** Alone, the tree model is *worse* than the regressions (+10%). Blended, the pair is clearly *better*, because they make different mistakes: the regressions are good at next week, the trees are good at seeing a turn coming 3-4 weeks ahead.

## Sentinel v3: three different kinds of guessers, equal say (October 7, 2026, `models/qar.py`)

v3 keeps everything in v2 and adds a third, different kind of model, then gives the three kinds an **equal say**:

| Model family | What it assumes | When it shines | When it fails |
|---|---|---|---|
| The 8 regressions | "the recent growth continues, bending a little" | steady, regular seasons (2024-25) | sharp spikes that suddenly stop (2025-26: +200% too high at the tip) |
| The trees (`models/gbq.py`) | "this season will turn like the ~15 past seasons did" | seeing a turn coming 3-4 weeks ahead (2025-26) | record seasons that go higher than anything in the past (2024-25) |
| **New: the quantile autoregression (`models/qar.py`)** | "next weeks' level depends on the last 3 weeks' levels, the way it did at this time of year in past years" | in between: rarely the best, rarely the worst | its ranges are too wide (its 95% range held the truth 98% of the time) |

**The new model** copies the method CMU published for its "CMU-TimeSeries" entry (the best team in 2025-26): it uses the last 3 weeks' levels (on the same common scale as the trees, so the 10 seasons of history count too), learns only from the same time of year (±3 weeks, every past year), and fits **each of the 23 quantiles directly** (quantile regression) instead of building ranges from past errors.

**Why "equal say" and not a tuned mix?** Because tuning failed honestly: when the mix was chosen on two seasons, it did badly on the third, every time. The seasons disagree about which family is best (see the table above), so any tuned weight is mostly luck. Equal weights have nothing to tune.

**The honesty label:** this choice was made *after* seeing all three seasons, so v3's scores are **in-sample**, not a blind test. v2's 0.789 on 2025-26 stays Sentinel's only blind result. The full story, including the rule this decision breaks and why, is Experiment 12 below and `backtest/V3_PROTOCOL.md`.

---

## Sentinel v3.2 and v3.3: two fixes found on October 30, 2026 (`models/engine.py`, rounds 24 and 26)
**v3.2, the QAR's "how low could it go" was far too long.** One of the three guessers (the QAR) gave a lower
end that the real number almost never reached: in past seasons the truth fell below its 2.5% line only 0.2% of
the time, instead of 2.5%. That dragged the whole team's range down. Fix: before the three guessers are
averaged, the QAR's lower half is pulled to 40% of its distance from its own middle guess. Past seasons:
-1.9% error (all three seasons better); this season's real forecasts: -2.0%.

**v3.3, a second opinion from ER visits.** In a small state the weekly admissions count is noisy: 5 one week,
9 the next, partly by luck. The share of ER visits that are flu rests on many more patients, so it shows more
reliably where the epidemic is. For each state under 200 admissions a week, Sentinel checks how many
admissions the newest ER share implies (using the state's own admissions-to-ER ratio of the 4 weeks before),
and moves the whole forecast half-way toward it. It only does this once flu is at least 1% of the state's ER
visits; below that the ER signal is too thin (tested live: it would have hurt in October 2026, when flu was
0.3-0.5% of ER visits in small states). Past seasons: -3.1% error on top of v3.2, all three seasons better.

**v3.4, the trees look at the neighbours.** The tree guesser now also sees the average level and recent slope of each state's bordering states (flu crosses state lines, and a neighbour that already turned is a clue). Past seasons: -0.9% error on top of v3.3, all three seasons better.

**v3.5, the trees also look at the whole country and the season so far.** The tree guesser now also knows the US average level and how fast it's changing, how the state compares with that average, the highest level the state has reached this season, how far below that it is now, and how many weeks ago that was. Past seasons: -1.7% error on top of v3.4, all three seasons better. The biggest help is in October (-7%), when a state that's still quiet can be read against a country that's already climbing. It's a little worse in January (+2.7%), the peak month.

**v3.6, the trees also look at ER visits.** The tree guesser now also sees how many ER visits are for flu in each state, and whether that share is rising faster than hospital admissions. ER visits come before hospital stays, so they're an early warning. Past seasons: -0.7% error on top of v3.5, all three seasons better. On the most recent weeks, where we have the exact ER data CDC published at the time, the gain is smaller (-0.35%) and not certain. If the ER file isn't updated on submission day, the trees automatically go back to v3.5.

How they were found: comparing this season's forecasts with the best teams' showed our middle guesses were
fine but our ranges in small states were about 20% wider than theirs. Each idea was written down in
`backtest/V3_PROTOCOL.md` before its result, chosen on past seasons only (leave one season out), and kept only
if it was better in every season with a 95% interval below zero. About a dozen other ideas tested the same day failed.

## Machine 4: the practice room (`backtest/`)

**The question:** "If Sentinel had existed in the past, how good would it have been?"

**How:** for each week of 3 past flu seasons (2023-24, 2024-25 and 2025-26), the program **pretends it's that week**:
1. It hides everything after that week, so no peeking at the future.
2. The guessers learn only from what was known before that week.
3. They guess 1, 2, 3 and 4 weeks ahead.
4. Then it un-hides the real numbers and **grades** each guess.

The v2 replay (`backtest/run_v2.py`) covers 79 forecast weeks and 15,790 situations (week × state × horizon), each guessed by the baseline, v1 and v2 on exactly the same footing: 47,370 graded guesses (`backtest/results_v2.csv`).

### The grading rule: WIS (`backtest/wis.py`)
WIS (weighted interval score) is the **exact grading rule the CDC uses**. In simple words:
- You lose points when the real number is **far from your guess**.
- You lose points when your range is **too wide** (being vague is penalized).
- You lose **extra** points when the real number falls **outside** your range (being overconfident is penalized hard).
- **Lower is better.** 0 would be a perfect guess.

**Relative WIS** = your score ÷ the baseline's score. Below 1 = better than the baseline. 0.70 = 30% better.

**Coverage:** "Did the real number land inside my 95% range?" Good guessers are right about 95% of the time. Sentinel v2: 94% (and its 50% range holds the truth 52% of the time). v3: 96% and 57%, a little too wide (its quantile autoregression part is the wide one). v1: 93%. The lazy baseline: only about 82%, so it's overconfident. By season, v2's 95% range held the truth 96%, 94% and 92% of the time: a little too narrow in 2025-26, the season with the record spike.

### Two kinds of practice: "corrected" and "real-time"
Hospital numbers get **corrected** after they're first published: about 1 time in 10, the first number is 18% or more too low. So there are two ways to practice:
- **Corrected mode** (`--mode revised`): use today's corrected numbers. Easier, so too optimistic.
- **Real-time mode** (`--mode realtime`): the CDC saved **97 versions** of the data, one per week. For each past week, Sentinel only sees the version that was **really published that week**, the same rough numbers the real teams had. **This is the fair test**, and the main results below use it. (10 weeks had no saved version and are skipped.)

In both modes, guesses are graded against the **final** numbers, exactly like the CDC grades the real teams.

### Comparison with the real CDC forecasts (`backtest/compare_cdc.py`)
The CDC publishes every past forecast from every team. So we took **the CDC's real baseline and the CDC's official ensemble** (the mix of all teams, which the CDC shows to the public) and graded them on **the same 15,590 guesses** as Sentinel's real-time replay (`backtest/vs_cdc_v2.md`).

| Score vs CDC baseline (lower = better) | 2023-24 | 2024-25 | 2025-26 | all 3 |
|---|---|---|---|---|
| Official CDC ensemble (live) | 0.740 | 0.657 | 0.669 | 0.677 |
| Sentinel v3 (replay, **in-sample**: chosen after seeing all 3 seasons) | 0.614 | 0.533 | 0.669 | 0.596 |
| **Sentinel v2** (replay; 2025-26 **blind**; re-run with the week-53 fix) | 0.651 | 0.519 | **0.789** | 0.638 |
| Sentinel v1 (replay) | 0.760 | 0.546 | 0.933 | 0.721 |
| CDC baseline | 1 | 1 | 1 | 1 |

(`backtest/vs_cdc_v3.md` has the v3 version of this table.)

**How to read this honestly:** 2023-24 and 2024-25 are the seasons every v2 decision was made on, so Sentinel's numbers there are flattered by that choice. 2025-26 is the only season v2 never looked at, and there **the CDC ensemble is still clearly better** (0.669 vs 0.789). The fair summary is: v2 is a big, real improvement on v1 (−15.5% on the unseen season), it beats the CDC baseline by about 21% on a season it never saw, and it is not yet as good as the CDC ensemble.

### The leaderboard (`backtest/leaderboard.py`)
We graded **every model from every team** in the hub (about 95 models: universities, Google, government labs) the same way, on the guesses they share with Sentinel. A model is ranked only if it made at least 75% of those guesses (the CDC's own rule uses 75%).

| Season | Sentinel v2's rank | Sentinel v3's rank (in-sample) | Best team that season |
|---|---|---|---|
| 2023-24 (design season) | 2nd of 28 | 2nd of 28 | UMass-flusion 0.603 |
| 2024-25 (design season) | 1st of 39 | 1st of 39 | Sentinel, then PSI-PROF 0.610 |
| 2025-26 (v2 never looked at it) | **23rd of 43** | 14th of 43 | CMU-TimeSeries 0.589 |

(`backtest/leaderboard_v2.md` and `backtest/leaderboard_v3.md` have the full tables.)

**Honest warnings:** Sentinel's guesses are a **replay**, not live submissions. Its design was chosen while looking at the first two seasons, which is why those ranks look so good. The live 2026-27 season is the real test.

### The leaderboard, the CDC's own way (`backtest/leaderboard_official.py`)
The table above divides each team's average score by the baseline's. The CDC ranks teams a bit differently, and the hub's rules say how:
- **Pairwise comparison** (Cramer et al. 2022): every team is compared with every other team on the forecasts both made, and its final score is the average (geometric mean) of all those head-to-head ratios. That's fairer when teams skip different weeks.
- **On the log scale**: before scoring, every number is turned into log(number + 1). Then "100 instead of 50" in a small state counts as much as "10,000 instead of 5,000" in California. On the plain count scale, California and Texas in the worst weeks of January decide almost everything.

| | v2, counts | v2, log (the CDC's ranking scale) | v3, log (in-sample) | CDC ensemble, log |
|---|---|---|---|---|
| 2023-24 | 2 of 28 | 2 of 28 | 1 of 28 | 0.718 |
| 2024-25 | 1 of 39 | 1 of 39 | 1 of 39 | 0.621 |
| 2025-26 (v2 blind) | 23 of 43 | **8 of 43 (0.610)** | 6 of 43 (0.594) | 0.616 |

**All 53 places, the exact live setup** (`backtest/eval_all53.py`, `backtest/all53.md`). The tables above cover the 50 states, but the CDC also scores the whole US, DC and Puerto Rico, and the live forecast trains on all 53 places. So the whole replay was run that way too. On the log scale, all 53 places, 2025-26: v3 0.592, v2 0.610 (blind), CDC ensemble 0.616. The whole-US forecast is one of Sentinel's best (v3 0.436 vs the CDC ensemble's 0.465 over 3 seasons); Puerto Rico is its weakest (0.854 vs 0.754).

So the honest blind result has two sides: on counts, v2 was clearly behind the CDC ensemble in 2025-26 (its January overshoot happened in the biggest states); on the log scale the CDC uses for rankings, it was level with it and 8th of 43.

### The "is it real or luck?" test (bootstrap)
A difference like "−1%" could be pure luck. To check, the computer **re-draws the past weeks at random 2,000 times** (some weeks twice, some not at all) and recomputes the difference each time. The middle 95% of those results is the **95% confidence interval**:
- If the interval is **entirely below 0**, the improvement is real.
- If it **includes 0**, we can't tell it apart from luck.

Whole weeks are drawn together, because all the states in one week share the same flu situation.

### The cheating bug we found (and fixed)
The first version said sewage made guesses **15% better**. That was too good to be true. The cause: the sewage cleaner compared each plant with its **whole history, including the future**. So the sewage number secretly "knew" whether a big wave was coming. After fixing it (past only, plus one week of delay), the honest answer became much smaller.

### So, does sewage help? (the research question)
Real-time replay, comparing models that differ **only** by sewage:

| | Change | 95% interval | Verdict |
|---|---|---|---|
| hospital trends → + sewage | −1.1% | −5.7% to +4.5% | no clear effect |
| hospital + ER visits → + sewage | −1.0% | −4.2% to +2.5% | no clear effect |

Sewage wins in about **58–59% of guesses**, but the average gain is tiny and could be luck. Honest conclusion so far: **sewage mostly tells us what the hospital numbers already say.** ER visits help a little (−2.4%, also not certain). The v1 ensemble as a whole was **32% better than Sentinel's own baseline** (interval −40% to −22%), which is clearly real. (These sewage numbers were measured with v1; v2 still keeps sewage as one of its three signals because it was pre-specified, not because it was proven.)

A small result is still a real scientific result. Reporting it honestly is the whole point.

---

## Machine 5: the packing machine (`submission/`)

### `build_submission.py`: making this week's file
1. **Reference date** = the next Saturday. That's how the CDC names each round.
2. **The last known week** = the Saturday before. That's the newest hospital data.
3. Run **Sentinel v3** (`models/engine.py`), the very same engine the practice room tests: fix the newest week's reporting delay (using the CDC's archive of past versions), add ER visits (same week when published), sewage (one week behind) and weather, fit the 8 regressions on everything known so far and average them, then take the equal-weight average with the tree model (`models/gbq.py`) and the quantile autoregression (`models/qar.py`). (`--model v2` builds the previous design instead, for comparison.)
4. Guess **horizons 0, 1, 2, 3**: the week ending on the reference date and the 3 weeks after.
5. Places without sewage data (the whole US, DC and Puerto Rico) automatically use only the regressions that don't need it.
6. Write the file with **exactly the 8 columns the CDC demands**:

| Column | Example | Meaning |
|---|---|---|
| `reference_date` | 2026-10-17 | Which week's round |
| `target` | wk inc flu hosp | "Weekly new flu hospital admissions" |
| `horizon` | 2 | How many weeks after the reference date |
| `target_end_date` | 2026-10-31 | The week being guessed |
| `location` | 48 | Texas |
| `output_type` | quantile | The kind of guess |
| `output_type_id` | 0.975 | Which of the 23 levels |
| `value` | 1238.4 | The guess |

The file is named like `2026-10-17-Sentinel-Ens.csv`, which is the CDC's naming rule. That's 53 places × 4 horizons × 23 numbers = **4,876 rows**.

**A second, optional target: the trend** (`submission/rate_change.py`). The CDC also asks "will admissions go *up a lot*, *up*, *stay stable*, go *down*, or *down a lot*?", as 5 probabilities that add up to 1, for each place and week. The thresholds are in admissions per 100,000 people (for next week: "stable" is a change of less than 0.3 per 100,000, or less than 10 patients; "a lot" is 1.7 per 100,000 or more; wider for later weeks). Sentinel doesn't need a new model for this: its 23 numbers already describe how likely every value is, so the probability of each category is read straight off them. Checked three ways:
- the category rules give exactly the CDC's own answers for every past week (32,767 of 32,767),
- in the replay, the trend probabilities scored better than the CDC ensemble's (Brier score 0.42 vs 0.51 in 2024-25, 0.42 vs 0.46 in 2025-26, lower = better; v3 is in-sample there, but this step has nothing tuned),
- their probabilities are honest: on average Sentinel said "stable" 43% of the time, and it was stable 43% of the time.

That adds 53 × 4 × 5 = 1,060 rows (`--no-rate-change` leaves them out). `python -m tests.test_submission_format` re-runs the checks.

### `validate.py`: the safety checker
Before anything leaves, it checks:
- exactly the right 8 columns, in the right order,
- one Saturday reference date, matching the file name,
- the dates line up with the horizons,
- every place code exists,
- **all 23 numbers** are there for every place and horizon,
- no negative numbers, and **no number goes down as the level goes up** (the 97.5% guess can't be lower than the 50% guess),
- **whole numbers only** (a hub rule since the 2025-26 season; the file builder rounds),
- for the trend target: exactly the 5 categories, probabilities between 0 and 1 that add up to 1,
- no guess above **30% of the place's population** (the hub's new "plausibility bound", added October 2026 to catch unit mistakes).

**If anything is wrong, it stops and nothing is sent.** The CDC also runs its own official checker on every file.

---

## Machine 6: the autopilot (`scripts/weekly_run.py` + `.github/workflows/weekly_forecast.yml`)

`weekly_run.py` does the whole week in one command: download (hospital, ER, sewage, weather and the CDC's archive of published versions), then clean sewage and weather, then build the file with v2, then check it, then **redraw the charts** (`scripts/plot_forecast.py`):
- `figures/latest_forecast.png`: this week's forecast for the US and 5 big states. A black line shows what was reported, a blue line Sentinel's best guess, and blue shading its 50% and 95% ranges.
- `figures/backtest_replay.png`: the replay of past seasons (Sentinel's 1-week-ahead guess vs what really happened).

The autopilot is a **GitHub Action**: a free robot that lives on GitHub's computers and wakes up **every Wednesday at 21:00 UTC** (5 PM in New York). The CDC's deadline is 11 PM New York time.

What it does each Wednesday:
1. Installs Python and the tools.
2. Runs `weekly_run.py`.
3. Saves the new forecast file **and the charts** in the Sentinel repository, so there's a public history of every guess.
4. **Only if the switch is ON:** copies the file into your fork of the CDC project and sends the CDC a **pull request** ("please add my guess"). This is the mailbox from before.

**The safety switch:** nothing goes to the CDC until you set `SUBMIT_TO_HUB = true` in the repository settings. You also need a **token** (a special password that lets the robot use your fork) saved as the secret `HUB_TOKEN`.

**If something breaks** (data late, a website down, the checker fails), the robot stops and **GitHub emails you**. A broken guess never goes out.

There's also a **"Run workflow" button** to start it by hand anytime.

---

## What's left before the first live forecast
**The 2026-27 FluSight season started on Wednesday October 7, 2026.** Every Wednesday, 11 PM New York time, is a deadline (reference date = the Saturday after). A missed week can't be sent late, but the CDC's evaluation only needs 75% of the weeks, so starting a week or two late is fine.

0. Upload the whole project to GitHub (the repository currently only has the README). Keep `data/history/` (the 10 seasons of history and the stored weather); `data/raw/` is re-downloaded every week and stays out.
1. Fill in your brother's name and email in `model-metadata/Sentinel-Ens.yml` (the two TODO lines; the methods text already describes v3 and passes the hub's schema check).
2. **Register the team:** add that file to the fork's `model-metadata/` folder and open a pull request to the CDC. That's the "team card". The first forecast file can go in the same pull request (`model-output/Sentinel-Ens/<reference date>-Sentinel-Ens.csv`).
3. Run `python scripts/weekly_run.py` once on your computer to see it work end to end (most of the time is the sewage download and the tree model).
4. Create the token, add the `HUB_TOKEN` secret, then switch `SUBMIT_TO_HUB` on.
5. Watch the first Wednesday run, then let it run by itself.

## Questions he should be able to answer in an interview
- **What does WIS measure?** Distance from the truth, plus penalties for vague or overconfident ranges.
- **Why the log scale?** So doubling looks the same whether a state has 100 patients or 10,000.
- **How did you avoid cheating in the backtest?** Each past week is replayed with the hospital data exactly as published that week; nothing after that week is used; sewage is scored against each plant's past only and used one week late; ER visits are used for the same week only where the CDC's archive shows they were published in time; every model is compared on identical forecasts; an automatic leak test re-runs forecasts with the future deleted and checks nothing changes. And design decisions were made on two seasons, with a third kept unseen as a final exam.
- **What's the biggest limitation?** Only one fully unseen season (2025-26), and on it Sentinel is behind the CDC ensemble. Sewage has no archived versions, so its replay is approximate. The live season is the real test.
- **What did you find about sewage?** No clear effect once hospital trends are known (−1%, with a 95% interval that includes 0), even though sewage tracks hospitals closely. Plus how a leak first made it look like 15%, and how he found and fixed it.
- **How do you know your backtest is fair?** It replays each past week with the data exactly as published that week (the CDC's weekly archive), grades against final data like the CDC does, and an independent reviewer re-ran the key results and found (and he fixed) three problems.
- **How good is it?** The only blind result: v2 on 2025-26. On the log scale the CDC uses for its rankings, 8th of 43 teams and level with the CDC ensemble (0.610 vs 0.616). On plain counts, behind it (0.789 vs 0.669, 23rd of 43), because one spike in the biggest states dominates count scores. 15.5% better than his own first version on that season. The submitted v3 looks better on that season (6th of 43 on the log scale), but v3 was chosen after seeing it, so that's not a fair test. The live season is.
- **What was the biggest single improvement?** Adding different kinds of models: gradient-boosted trees (after UMass's flusion) and a quantile autoregression (after CMU), both learning from 10 extra seasons of history. Each fails in different seasons; averaged, they cover for each other.
- **What's the most interesting thing you learned?** That the seasons disagree about which model is best, so tuning the mix on past seasons doesn't transfer: weights chosen on two seasons did worse on the third every time. And that he had to break one of his own rules to choose v3, and wrote down why.

---

## Experiment log (add every new clue here)

The rule: **test one new clue at a time**, with the fair replay and the "real or luck" test. Keep it in the final forecast **only if the 95% interval is entirely below 0**. Otherwise it stays out, and we write down the result anyway.

### Experiment 1: neighbouring states (October 5, 2026)
- **Idea:** flu spreads across borders, so if the states next to Texas are rising, Texas may rise next.
- **How:** a border list (`models/neighbors.py`: 48 mainland states plus DC, every border checked both ways). New clues `nb1` and `nb2` = the average of the neighbours' change last week and the week before, from the same data version as the state itself.
- **Result:** with hospital trends only, −1.6% (95% interval −8.5% to +6.9%). With everything else, −0.3% (−5.2% to +5.3%). **No clear effect, so it stays out** of the final forecast.
- **Why it probably didn't help:** a state's own recent trend already contains most of what the neighbours tell us. Flu tends to rise everywhere at about the same time.

### Experiment 2: lab-test positivity (October 5, 2026)
- **Idea:** the share of flu lab tests coming back positive (a CDC/Delphi clinical-labs number, separate from hospital admissions and wastewater) might catch a rise before it shows up in hospitals.
- **How:** pulled from CMU Delphi's public `fluview_clinical` data (44 states, weekly, back to late 2021), turned into one weekly log-scale number per state (`scripts/lab_clean.py`), used with the same 1-week delay as the other outside signals so nothing from the future leaks in.
- **Result:** with hospital trends only, this one is real: −1.8% (95% interval −2.9% to −0.6%, better in 67% of weeks) — the interval is entirely below zero. But stacked on top of ER visits and wastewater (which already track the same rise), the extra benefit vanishes: −0.4% (−1.5% to +0.6%) with everything, and −0.4% (−1.1% to +0.1%) with neighbours added too. **Stays out** of the final forecast — not because it's fake, but because ER visits and wastewater already capture what it would add.
- **Why:** lab positivity is an early-warning number, but so is the ER-visit share of flu — they're telling us roughly the same thing a different way, so once one is in, the other adds little.

### Experiment 3: weather (October 5, 2026)
- **Idea:** cold, dry air is strongly linked to flu spreading. The model already knows the calendar date (the sin/cos terms), but actual weather can differ from the calendar: an unusually warm or wet week, or a state that's colder/drier than normal for other reasons.
- **How:** daily temperature and humidity per state (Open-Meteo, free, no key needed), turned into one weekly "colder and drier than this state's own past" number (`scripts/weather_clean.py`) — standardised against each state's past only (expanding window, at least 52 weeks seen first), the same past-only rule as wastewater, so nothing from the future leaks in.
- **Result:** with hospital trends only, −2.7% (95% interval −5.5% to −0.1%). With ER visits and wastewater already in, it got *stronger*, not weaker: −4.0% (−6.8% to −1.4%). **Both intervals are entirely below zero — a real effect, not luck.**
- **Why it's different from lab positivity and neighbours:** those two were measuring roughly the same thing the clinical/sewage data already measures (people getting sick/tested), so they became redundant once ER visits and wastewater were in. Weather is a genuinely separate channel — actual local conditions, not another read on how many people are already sick — so it adds information the others don't have.
- **Decision:** promoted into the submitted ensemble. `ENSEMBLE_SIGNALS` in `models/models.py` is now `("ed", "ww", "weather")`, and the ensemble is every combination of the three (8 models, up from 4). Overall result: Sentinel's relative WIS vs the CDC's real forecasts improved from 0.718 to **0.700** (CDC's own ensemble: 0.680) — the gap closed by about half.

### Experiment 4: why does the model do badly some weeks, and can we fix it? (October 5, 2026)
- **The question:** 2025-26 is Sentinel's worst season by far (rank 24 of 43, vs. 1st and 5th the other two). Why?
- **Digging in:** broken down by week, the whole season is fine except three weeks in January 2026, where the model was actually *worse than the naive baseline* — something that never happens any other week, in any season. Those three weeks line up exactly with a real event: US flu hospital admissions went 37,505 → 42,505 → 29,912 → 19,735 in three weeks, a very sharp, fast peak and crash. The model, trained on the past few weeks of steep rise, kept predicting more rise right as the real number collapsed (for one state, it guessed about 4,750 when the true number was 1,233).
- **First guess (wrong):** maybe the model was just too *confident* -- too narrow a range around its guess, so it got penalised hard for being surprised. Tried making the range wider automatically whenever a state's recent growth has been unusually jumpy. Barely changed anything (−1.5% overall, could be luck, 95% interval −6.3% to +4.3%). **Lesson: the problem wasn't the range, it was the guess itself being 4x too high.**
- **Second attempt:** gave the model two new clues so it can learn "a very fast rise, especially when the number is already way above normal for this time of year, tends to bend back down soon" instead of assuming it always keeps rising in a straight line. Fixed one of the three bad weeks completely. Couldn't fix the worst one (the forecast made right at the tip of the peak, before any clue could have seen the crash coming -- that one really was a blind guess with no way to know).
- **Honest result:** −1.5% overall change, 95% interval −6.3% to +4.4% -- **not statistically proven**, could be luck. Kept it anyway since it's free (no new data source, so no new way to leak the future) and it provably fixed one real failure without creating new ones. Combined with the weather signal, Sentinel's score vs. the CDC's real forecasts moved from 0.718 (where this session started) to **0.690** -- the CDC's own ensemble scores 0.680, so the gap that was 4 percentage points is now 1.
- **What's NOT fixed:** a forecast made right at the exact top of an unprecedented, never-seen-before spike is close to unpredictable from past data alone -- there was no clue yet that it was about to crash. No amount of feature engineering fixes a blind one-step shock; only a model that tracks *why* flu is behaving unusually that year (an unusually severe strain, vaccine mismatch) could have a shot, and that's a much bigger project than this experiment.
- **Calibration check (is the model's range of guesses honest, not just the single number?):** the 50% and 95% ranges hit their targets closely in every season (50% range covers the truth 45-52% of the time, 95% range covers it 93-94% of the time) -- 2025-26 is actually the *middle* of the three seasons, not the outlier. This confirms the problem really was those three bad weeks' point guess, not the model being over- or under-confident everywhere.

### Experiment 5: fixing the opposite problem -- undershooting a new season's climb (October 6, 2026)
- **The complaint, fair and direct:** checked against the last 3 months of real data (this year's season just starting to climb), Sentinel's 1-week-ahead national guess was consistently LOW -- off by 15-25%, every single week from July through September 2026. Same root cause as the January whiplash, mirrored: a trend-following model lags behind any sharp CHANGE in trend, whether the real curve is turning down (overshoot) or speeding up (undershoot).
- **First attempt (made it worse, kept for the record):** added a correction that checks each state's own last 4 weeks of guesses against what actually happened, and nudges the current forecast by the full size of that recent error. Fixed the live undershoot nicely (errors dropped to roughly ±0-15%). But run against the full 3-year history, it was a clear regression: the ensemble's advantage over the baseline collapsed from -34% to -15%, worse in **all three seasons**. Chasing a short 4-week window amplifies noise and actively hurts at every turning point, which is most of history -- it was overfitting to exactly the one pattern (steady climb) seen in the most recent few months, at the cost of everything else.
- **Fixed version:** same idea, but only apply 30% of the measured recent bias (`BIAS_SHRINK = 0.3`), over a 6-week window instead of 4, capped so it can't run away. Checked both ways: historical 3-year backtest now shows +4.2% (95% CI -3.8% to +12.8% -- crosses zero, not a proven regression, essentially neutral), while the live 3-month undershoot improves from an average 15.5% error to about 11.7%, and the worst week stops being systematically on one side.
- **Honest limits:** this doesn't fully fix the lag, it softens it. A model that reacts instantly to the latest week would overfit to noise; a model that never reacts is stuck with the bias just measured. The 30% shrink is a judgment call, not something separately proven optimal -- picked because the alternative (0% or 100%) was tested and was worse at one end or the other.

### Experiment 6: a genuinely different model type, not just another linear variant (October 6, 2026)
- **The idea:** every model so far is a flavor of the same thing -- linear regression on the log scale. The biggest structural gap vs. serious teams (CMU, UMass, Google) is that they combine *different kinds* of models (mechanistic epidemic models, machine-learning models, statistical ones), so when one type is wrong the others often aren't. Tried adding a gradient-boosted decision-tree model (`HistGradientBoostingRegressor`, scikit-learn) alongside the linear one -- trees can learn nonlinear patterns and interactions automatically, instead of needing them hand-built like the peak-dampening features.
- **Tested the right way from the start:** checked using ONLY the two training seasons (2023-24, 2024-25), never touching 2025-26, following Experiment 5's lesson.
- **Honest result: it's worse.** 22.5% worse than the single best linear model (95% CI +7.2% to +37.2% -- entirely positive, a real, confident loss, not noise). Tried regularizing it harder (shallower trees, bigger minimum leaf size, an L2 penalty) in case it was just overfitting -- that helped a little (95.0 vs 96.4 mean WIS on one season) but still clearly behind the linear ensemble (91.4) and the single best linear model (87.6) on the exact same weeks.
- **Why:** the real relationship here (next week's log-change ~ recent log-changes, time of year, extra signals) is close enough to linear that a tree model, trained on only a few thousand pooled rows per fit, can't find anything useful beyond what the linear model (plus the hand-built nonlinear features) already captures. More data or a fundamentally different kind of model (an actual epidemic-mechanism model, not a generic ML one) is a bigger project than swapping one algorithm.
- **Decision:** NOT added to the submitted ensemble. Kept in the code (`fit_and_predict_gbm` in `models/models.py`) and still scored every run for the record, so if more years of data become available this can be retried honestly instead of guessed about.

### Experiment 7: test every idea, one at a time, the fast way (October 6-7, 2026)
- **New rule for everything from here on:** decide using **2023-24 and 2024-25 only** ("design seasons"). 2025-26 is kept as the **final exam**, scored once at the very end.
- **New tool:** `backtest/harness.py` replays the design seasons in about a minute per idea (the full backtest took 10+). Checked first: on the same forecasts it gives the same scores as the official backtest, row for row.
- **What each idea did on its own** (change in score vs. the v1 design; "real" = the 95% range is entirely below 0):

| Idea | Change | Verdict |
|---|---|---|
| Fix the newest week's reporting delay | -0.2% | too small to tell (kept anyway: it makes the data more correct) |
| ER visits for the same week (with the ~2.8% first-publication adjustment) | -2.9% | almost real |
| Recent weeks count more (half-life 52 weeks) | -1.9% | almost real |
| "Already infected this season" clues (`cum`, `cum_d1`) | -0.3% | real, but small |
| Count big states more when fitting | +1.4 to +3.2% | worse |
| National growth as a clue | +9.2% | worse |
| Wider ranges (x1.2, x1.4) | +2 to +6% | worse (ranges were already right: 50% range held the truth 50% of the time) |
| Linear quantile regression | too slow to finish | dropped |

- **Mistake caught on the way:** a column was accidentally used twice in the first version of the harness; it crashed instead of giving wrong numbers, and was fixed before any result was used.

### Experiment 8: all the good ideas together (October 6, 2026)
- Same-week ER + recent-weeks-count-more + "already infected" clues + reporting-delay fix, together: **-4.5% (95% range -7.6% to -1.6%) -- real.**
- Tried on top of it: no bias correction (+1.4%, so the bias correction stays), stronger delay fix, no volatility widening, narrower ranges, ranges by size: all within +/-1%, nothing real. This combination became the "linear part" of v2.

### Experiment 9: teach the regressions 10 more seasons of history (October 7, 2026)
- **Data:** CDC FluSurv-NET hospitalisation rates (14 sites, 2010-2020) and ILI+ for 49 states (2010-2020), from UMass's public flusion repository (`scripts/build_history.py`). Only data from before July 2022, so it can't know anything about the seasons being tested. The 2009 pandemic and the two COVID seasons were dropped.
- **Mistake caught before it counted:** the first version of this test accidentally switched off "recent weeks count more" at the same time, which would have mixed two changes. Stopped, fixed, re-run.
- **Result:** six versions (different weights, only one source, only as an extra member): all between -0.6% and +1.3%, **no real effect**. Makes sense: a straight-line model has only ~15 numbers to learn, and 3 years of data already pin them down. More data doesn't help a model that can't use it.

### Experiment 10: a tree model that CAN use the history (October 7, 2026)
- Built `models/gbq.py`, a smaller version of UMass's GBQ model: gradient-boosted trees trained on the hospital data plus all the history, on a common scale, bagged over seasons.
- **Results on the design seasons** (score relative to the baseline; lower is better):

| Tree model weight in the blend | Score | vs. v2 linear part |
|---|---|---|
| 0% (linear only) | 0.554 | - |
| 20% | 0.529 | -4.5% (real) |
| **33%** | **0.522** | **-5.9% (real: -11.9% to -0.6%)** |
| 50% | 0.522 | -5.9% (not sure) |
| 100% (trees alone) | 0.610 | +10% (worse) |

- Same with 3 or 6 bags (-6.2% vs -5.9%), so it's not luck in the trees. 6 bags kept for stability.
- Note: these numbers were measured before the independent review fixed the ER-data timing (same-week ER assumed in 2023-25). After the fix, on the same design seasons, the 33% blend vs the linear part alone is still a gain in 2023-24 (0.731 -> 0.624 vs Sentinel's own baseline) and about even in 2024-25 (0.496 -> 0.492).
- Where it helps: a lot in 2023-24 (0.704 -> 0.602), about nothing in 2024-25; little for next week, a lot for 3-4 weeks ahead -- exactly where knowing how seasons *turn* matters.

### Experiment 11: ranges sized per place (October 7, 2026)
- Found by looking at the live national forecast: its 95% range said US admissions could halve in one week, which never happens. Cause: one shared error size for all places.
- Each place's range moved halfway towards its own past error size: **-0.5% on the 50 states (95% range -1.0% to -0.1%) -- real**, and it fixes the national range, which the 50-state score doesn't even include.

### The final exam: 2025-26, scored once (October 7, 2026)
After the design was locked (Experiments 7-11), v2 was scored on 2025-26, which no v2 decision had looked at:

| | v1 | v2 | CDC ensemble (live) |
|---|---|---|---|
| 2025-26, vs the CDC baseline | 0.933 | **0.789** | 0.669 |
| v2 vs v1 (95% range) | | **-15.5% (-18.9% to -9.7%) -- real** | |

- The gain on the unseen season (-15.5%) is bigger than on the design seasons (-8.5%): the opposite of what overfitting looks like.
- **But Sentinel is still behind the CDC ensemble on this season, and ranks 23 of 43 teams.**
- Two forecast weeks (3 and 10 January 2026, at the tip of the record spike) make up 40% of v2's error for the whole season. Every other week, v2 beats the baseline by 20-60%. Not tuned away on purpose: that would turn the exam into homework.

### The independent review (October 7, 2026)
A separate review, done fresh without the design notes, went through all the code looking for anything that makes the results look better than they are. It reproduced the held-out result exactly, re-fitted one tree-model week bit-for-bit, and checked the scoring, the horizons, the truth data and the pairing: all clean. It found three real problems, all fixed:

1. **ER data timing (the important one).** The practice room assumed same-week ER numbers in 2023-25, but back then they came out two days after the deadline. **Fix:** same-week ER only where the CDC's archive shows it was published that way. Cost: the design seasons got about 2-3% worse (2023-24: 0.605 -> 0.623; 2024-25: 0.482 -> 0.491 vs. Sentinel's own baseline); the held-out season didn't change (most of its weeks have the real archived ER numbers). The "-2.9%" for same-week ER in Experiment 7 was measured under the wrong assumption and is overstated.
2. **A small cheating bug in gap-filling.** When a wastewater value was missing, the gap was filled *before* applying the one-week delay, which could pull in a value one week early: 8 forecasts out of about 4,000. **Fix:** delay first, then fill gaps, in both v1 and v2.
3. **The leak test had blind spots:** it used final hospital numbers instead of the archived ones, and it could not notice a signal being used too early. **Fix:** it now uses the archived hospital data, checks that deleting the newest wastewater week changes nothing, and checks the ER-availability rule.

It also made a fair point about honesty: the "beats the CDC ensemble over 3 seasons" claim (0.637 vs 0.677) is mostly the design seasons, where many experiments were chosen. On the one fully fair season, the CDC ensemble is still better.

### Experiment 12: Sentinel v3, after the exam (October 7, 2026, `backtest/V3_PROTOCOL.md`)
- **The problem:** the final exam was used up, so there is no unseen past season left. Rules were written down *before* any v3 result: ideas listed in advance; any number to choose (like a weight) chosen on two seasons and scored on the third ("leave-one-season-out"); keep an idea only if it's better overall (interval below 0) and not worse in any season.
- **Six pre-listed ideas, none passed.** Tree weight chosen per season or per horizon, median of the regressions, a third model (quantile autoregression, CMU's published method), two training windows: all between -2.3% and +0.0%, intervals crossing 0. The weights chosen on two seasons kept failing on the third.
- **Why:** the model families win in different seasons (regressions in 2024-25, trees in 2025-26), so the "best mix" on two seasons is wrong for the third.
- **Two follow-ups, declared before running them:** equal weights for the three families (-6.5%, interval -13.5% to +0.5%; 2023-24 -5.6%, 2024-25 +2.8%, 2025-26 -15.0%) and weights that adapt during the season (no better than equal weights).
- **Decision:** equal weights, adopted as an openly declared judgment call: it narrowly fails his own rule (2024-25 is 2.8% worse, and the interval touches +0.5%). On the two original design seasons it ties v2 (-0.2%); its advantage comes from 2025-26. All v3 numbers are labelled in-sample.
- **After the second review** (next section), two more ideas written down before their results: no QAR for Puerto Rico (QAR assumes mainland flu seasons; kept: Puerto Rico 0.936 -> 0.854 vs the CDC baseline, log scale) and one LOSO-chosen spread factor for the ranges (not kept: +0.9%).
- **Checks:** the v3 results reproduce v2's official file exactly before adding anything; the engine run end-to-end gives bit-identical forecasts to the replay caches; the leak test passes on v3 (see below).

### Where v3 is weaker (October 7, 2026)
v3 is not better everywhere. Month by month (log scale, all 53 places), it beats v2 around the peak and the turn of a season, and loses in the tails: at the end of a season (April-May) it is worse in 5 of 6 months, and at the start it is mixed. The trees and the quantile autoregression learned from ~10 older seasons that flu fades fast after the peak; in the CDC's hospital data it fades more slowly. In July-September 2026 (`backtest/recent_check.md`, never used for any decision), v3 was also worse than v2 and under-predicted the early rise of the US more (1-week-ahead medians about 10% too low on average, v2 about 8%). This season started unusually early. Nothing was changed because of this (a change chosen now would be tuned on a few months), but it's the first thing to re-check once 6-8 weeks of the live season are in.

### Experiment 13: hunting for what still goes wrong (October 7, evening; `backtest/V3_PROTOCOL.md` rounds 3-5)
- **New main score:** from here on, ideas are judged the way the CDC ranks teams: log scale, all 53 places.
- **Eleven ideas, written down before their results, failed:** a fourth model (a simple AR(6)), weighting the hospital data more, narrower ranges for small counts, correcting the low bias from recent weeks, a fixed upward shift, last week's CDC ensemble as an extra voice (so Sentinel stays independent of the other teams), absolute humidity instead of the cold/dry index, the QAR trained on all weeks, the QAR with a season term, no QAR at all.
- **What the failures taught:** v3 forecasts too low more often than too high (the truth is above its middle forecast 58% of the time), but how much changes from season to season, so no fixed correction works. And the QAR needs its "same time of year" window, even though that window is exactly what makes it blind to an early season like this one.
- **The bug that was found (v3.1):** the first live week showed v3 under-forecasting the rise more than v2. Taking the forecast apart showed why: the tree model was fed the newest week as first published (typically 7-9% too low), while the regressions got the corrected number. Fixing that: 0.6% better on the CDC's scale, in every season. Small, but real, and it removes a known cause of the low bias.

### The second independent review: v3 (October 7, 2026)
Another fresh reviewer re-checked everything about v3 without having seen the work. Every number it recomputed matched. What it found, and what was done:
1. **The week-53 calendar bug.** "Flu season" was worked out from the calendar year. But a few January Saturdays belong to the last week (52 or 53) of the *previous* year in the ISO calendar the CDC uses, e.g. 2 January 2016 and 2 January 2027. Those days landed in the wrong season, which would have reset "admissions so far this season" to almost zero right at the January 2027 peak. In the replay it only shifted the week numbers of one history season (2015-16) for the trees and the quantile autoregression; the regressions were unaffected (the hospital data start in February 2022). **Fix:** one shared, correct season calendar (`models/seasons.py`) for all three model families, and **every replay was re-run** with it; all results now come from the fixed code.
2. **The US, DC and Puerto Rico were submitted but never scored.** A first look showed Puerto Rico badly forecast by the quantile autoregression, which assumes mainland seasons. Now all 53 places are replayed in the exact live setup (`backtest/eval_all53.py`, results in `backtest/all53.md`).
3. **The trend probabilities** counted a change of exactly 10 patients as "stable". Fixed (whole-number boundaries).
4. **Hub rules:** the validator now also checks that the week is a real round, the target weeks are allowed, and today is inside the Sunday-Wednesday window (US Eastern time); the autopilot skips weeks that are not rounds and only ever submits this week's file.
5. Smaller things: the weather download restarts per place, archive weather wins over forecast-service weather, the wastewater step warns about stale data, a place or model family that drops out is now reported instead of silently skipped, and some wording that said more than the evidence (fixed in the README).

### Honesty check: does the design actually work on a season it never saw? (October 5, 2026)
- **The worry (raised directly):** every decision above (which signals to keep, which features to add) was made by looking at results across all three seasons, including 2025-26 -- the same season used to judge whether those decisions worked. That's circular. A real forecasting team has to commit to their design *before* the season happens.
- **The test:** redid every decision -- which signals to keep (ed, ww, weather), the dampening features, how many weeks of history to train on -- using ONLY 2023-24 and 2024-25. 2025-26 was never looked at while deciding anything. Then ran the locked design on 2025-26 exactly once, as a genuinely blind test.
- **What this caught:** the "longer training window helps" finding from earlier was accidentally cheating -- there isn't enough historical data for a longer window to even make a difference until you get to 2025-26, so testing it on 2023-24+2024-25 gave exactly 0%, no signal at all. The entire apparent improvement was coming from the one season it was being tested against. **Reverted to the original 156-week window.**
- **What held up:** weather, ER visits and wastewater all still showed a real effect using only the two training seasons (weather on top of everything: −5.6%, 95% CI −9.4% to −1.4%, still entirely negative). Neighbours and lab positivity still didn't earn a place. The dampening feature's effect on the two training seasons alone was −1.3% (95% CI −5.4% to +3.8%, not significant) -- unproven, but it doesn't disappear or reverse on seasons it wasn't designed around, so it was kept.
- **The honest result:** with everything decided blind, the locked design scores 0.794 relative WIS on 2025-26 -- almost identical to the 0.793 reported before this check. So most of the earlier numbers hold up; only the training-window change was actually hindsight, and it's now fixed.

### Experiment log, status
Neighbouring states and lab-test positivity were tested and kept OUT of the submitted ensemble (real on their own, redundant once better signals are in). Weather was tested and kept IN — it's now pre-specified alongside ER visits and wastewater in `ENSEMBLE_SIGNALS`. The peak-dampening features (`d1sq`, `d1_rel`) are now permanent base features (not gated behind an experimental signal) since they use no new data and passed the leak test cleanly.

### The leak test (`tests/test_no_future_leak.py`)
**The worry:** a program can "see the answer" in sneaky ways and look better than it really is.

**The test:** make the forecasts for a past week **twice**: once with all the data, once with **everything after that week deleted**. If even one number changes, something is peeking at the future.

**What it caught (October 5, 2026):** it failed at first. The program chose *which states to include* based on whether a state has sewage data **at any time, including the future**. States that only started testing sewage later were included in the past too. No future flu numbers were used, but the choice of states depended on the future. **Fix:** always use a fixed list of the 50 states. The test now **passes** on 3 test weeks (about 3,400 forecasts identical), and the results barely changed (Sentinel 0.718 vs the CDC baseline, same ranks).

**Run it after every change:** `python -m tests.test_no_future_leak`. Re-run after adding lab-test positivity (still passes, 0 differences across all three test weeks) — adding a new signal doesn't automatically introduce a leak, but it's cheap to check every time.

### The leak test for the engine that runs now (`tests/test_engine_no_future_leak.py`)
The same idea, made stricter after the independent review, for the exact code that sends the forecasts:
1. **Future check:** forecast a past week with everything, then again with everything published after it deleted (later hospital numbers, later signals, later versions in the CDC's archive). The hospital data used is the version **as published that week**, not today's corrected one.
2. **Wastewater-delay check:** wastewater is used one week behind, so deleting the wastewater of the forecast week itself must change nothing. (This is the check that would have caught the gap-filling bug.)
3. **ER check:** when the newest ER numbers are older than the forecast week (as before 2025-26), the engine must behave exactly as with ER one week behind.

Run on 3 weeks (one per season), for the 50 states and for all 53 places, with all three model families, plus the long-history option and the ER rule. **v3.1 (what is submitted): PASS, 0 differences in every check** (`tests/results/leak_engine_v31.log`); v3 and v2 also passed (`leak_engine_v3.log`, `leak_engine_v2.log`). It takes about an hour because the tree model is re-trained for every check.
