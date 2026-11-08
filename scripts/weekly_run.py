"""The whole weekly job in one command (this is what the autopilot runs every Wednesday).

    python scripts/weekly_run.py

1. download the latest hospital, ER, wastewater and weather data + CDC's archive of published versions
2. clean the wastewater and weather into one weekly number per state
   (if the coming Saturday is not one of the hub's forecast rounds, e.g. in summer, stop here)
3. build this week's forecast file (Sentinel v3, models/engine.py)
4. validate it (stops with an error if anything is wrong, so nothing broken is ever submitted)
5. build the v2 shadow forecast (never submitted) and update the live scoreboard
6. redraw the charts shown on the GitHub page
"""
import subprocess
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]):
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    import os
    py = sys.executable
    run([py, "scripts/ingest.py"])
    sys.path.insert(0, str(ROOT))
    from submission import hub_rules
    from submission.build_submission import TEAM_MODEL, next_saturday
    ref = next_saturday(date.today()).isoformat()
    tasks = hub_rules.load_tasks()
    if tasks is not None and ref not in hub_rules.reference_dates(tasks):
        print(f"\n{ref} is not a FluSight forecast round (off-season): nothing to build this week.")
        sys.exit(0)
    run([py, "scripts/wastewater_clean.py"])
    run([py, "scripts/weather_clean.py"])
    run([py, "-m", "submission.build_submission"])
    check = [py, "-m", "submission.validate", f"submission/output/{ref}-{TEAM_MODEL}.csv"]
    if os.environ.get("SUBMIT_TO_HUB") == "true":
        check.append("--for-submission")  # it's going to the CDC: also check the round and the deadline window
    run(check)
    # v2 shadow forecast (never submitted): kept to compare v2 and v3 on the live season
    run([py, "-m", "submission.build_submission", "--model", "v2", "--out-dir", "submission/shadow/v2"])
    subprocess.run([py, "-m", "backtest.score_live"], cwd=ROOT)  # live scoreboard (never blocks the run)
    run([py, "scripts/plot_forecast.py"])
    print(f"\nDone. File ready: submission/output/{ref}-{TEAM_MODEL}.csv")
