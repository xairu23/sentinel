# Submitting a forecast by hand (GitHub website, works from a phone)

Use this until the autopilot (`.github/workflows/weekly_forecast.yml`) is switched on.
Deadline: **every Wednesday, 11 PM New York time**: Thursday 03:00 UTC until November 1, then
Thursday 04:00 UTC until March 14, 2027, then 03:00 UTC again. Check your phone's clock against UTC:
in October 2026 Moroccan time is UTC+0, so tonight's deadline is **03:00 Thursday, Moroccan time**.
Late files are not accepted.
The file for a week is named `<reference date>-Sentinel-Ens.csv`, where the reference date is the
Saturday after that Wednesday (first one of the season: `2026-10-10-Sentinel-Ens.csv`).

## The first time: team card + forecast in one pull request

1. **Fork is up to date.** Open `https://github.com/xairu23/FluSight-forecast-hub`. If GitHub shows
   "This branch is N commits behind cdcepi:main", press **Sync fork → Update branch**.
   (No fork yet? Open `https://github.com/cdcepi/FluSight-forecast-hub` and press **Fork**.)
2. **Team card.** Open
   `https://github.com/xairu23/FluSight-forecast-hub/new/main/model-metadata?filename=Sentinel-Ens.yml`,
   paste the whole content of `model-metadata/Sentinel-Ens.yml` from this project, and replace the two
   `TODO` lines with the real name and email. At the bottom choose **"Create a new branch for this
   commit and start a pull request"**, name the branch `sentinel-first`, press **Propose changes**.
   **Don't open the pull request yet** (close that page).
3. **Forecast file.** Open
   `https://github.com/xairu23/FluSight-forecast-hub/upload/sentinel-first/model-output/Sentinel-Ens`,
   upload the week's CSV, choose **"Commit directly to the sentinel-first branch"**, press **Commit changes**.
4. **Pull request.** Open `https://github.com/cdcepi/FluSight-forecast-hub/compare/main...xairu23:FluSight-forecast-hub:sentinel-first`,
   check it shows exactly 2 files (the `.yml` and the `.csv`), title it `Sentinel-Ens: team metadata and first forecast`,
   press **Create pull request**.
5. **Watch the checks.** The hub's robot validates the files within a few minutes and comments on
   the pull request. Green = done; the CDC team merges it. Red = read the comment (usually a typo in
   the `.yml`) and fix the file on the same branch; the checks re-run by themselves.

## Every following week (until the autopilot is on)

1. Make the file: `python scripts/weekly_run.py` (after the CDC's Wednesday data update).
2. Open `https://github.com/xairu23/FluSight-forecast-hub/upload/main/model-output/Sentinel-Ens`
   (sync the fork first), upload the new CSV, choose **"Create a new branch ... and start a pull request"**,
   then make sure the pull request goes to **cdcepi/FluSight-forecast-hub main**.

## Before sending, always

`python -m submission.validate submission/output/<file>.csv --for-submission` must print `VALID`. It checks
the hub's format, the 23 quantiles, whole numbers, the trend probabilities, the hub's plausibility bound,
and that the week is a real round and today is inside its submission window (Sunday to Wednesday).
