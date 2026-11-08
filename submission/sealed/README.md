# Sealed forecasts: made before the truth was published

These files were made on **Wednesday October 7, 2026 at 02:56 UTC (v3) and 02:59 UTC (v2)**, from data
up to the week ending September 26, 2026 (the latest the CDC had published). The CDC publishes the
week ending October 3 later on October 7, and the following weeks after that. Neither version has
seen any of the weeks it forecasts, so these are the first fully fair tests of v3, and v2 keeps
being tested alongside it.

Reference date 2026-10-03, horizons 0-3 = weeks ending October 3, 10, 17 and 24. (2026-10-03 is not
one of the hub's rounds, the season's first is 2026-10-10, so these files are for checking only and are
never submitted. They hold the weekly-admissions quantiles only: the trend target was added later the
same night. They were made with the code as it was then, before the week-53 calendar fix, which
changes a few training rows slightly; they are kept exactly as made.)

| File | Model | SHA-256 |
|---|---|---|
| `2026-10-03-Sentinel-Ens.csv` | v3 as it was at 02:56 UTC (before the Puerto Rico rule and the calendar fix) | `12c05651de441bf092b5e08c882d466eac8a284e844ef7b9e21953a77dc94678` |
| `v2/2026-10-03-Sentinel-Ens.csv` | v2 (previous design) | `db2d9c3938214d9a6d653446eee0cebc92427c0a5e73e18bb92c30b9ffc3ee92` |

Whole US, patients admitted with flu (last known week, September 26: 2,515):

| Week ending | v3 median | v3 95% range | v2 median | v2 95% range |
|---|---|---|---|---|
| Oct 3 | 2,860 | 1,098 - 5,946 | 3,124 | 1,638 - 5,611 |
| Oct 10 | 3,234 | 1,105 - 8,063 | 3,697 | 1,655 - 7,770 |
| Oct 17 | 3,661 | 1,118 - 10,385 | 4,311 | 1,682 - 9,936 |
| Oct 24 | 4,263 | 1,140 - 13,880 | 5,147 | 1,725 - 13,082 |

To check them once the data is out: score both files against `target-data/target-hospital-admissions.csv`
with `backtest/wis.py` (or just compare the medians above with the published numbers).

## Result for the week ending October 3 (published October 7, 2026)
Whole US: **actual 3,246**. v2 median 3,124 (-3.8%), v3 median 2,860 (-11.9%), flat baseline 2,515
(-22.5%). v2's and v3's 95% ranges each held the truth in 52 of 53 places. Mean WIS over the 53 places, log
scale (lower = better): v2 0.218, v3 0.231. Both clearly beat the baseline; v2 beat v3. Taking v3
apart showed why it was lower, and led to the v3.1 fix (the trees now get the corrected newest week;
`backtest/V3_PROTOCOL.md`, idea 19). Later weeks: `backtest/live_scoreboard.md`
(`python -m backtest.score_live`).
