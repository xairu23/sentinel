"""The CDC hub's own round rules (hub-config/tasks.json): which Saturdays are forecast rounds, which
target weeks are allowed, and when a round's file may be submitted (Sunday to Wednesday before the
reference date, US Eastern time). scripts/ingest.py downloads tasks.json to data/raw/hub_tasks.json.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = (ROOT / "data/raw/hub_tasks.json", ROOT.parent / "FluSight-forecast-hub/hub-config/tasks.json")


def load_tasks() -> dict | None:
    for p in CANDIDATES:
        if p.exists():
            return json.loads(p.read_text())
    return None


def _ids(task_ids: dict, key: str) -> set:
    v = task_ids.get(key, {})
    return set((v.get("required") or []) + (v.get("optional") or []))


def reference_dates(tasks: dict) -> set:
    return set().union(*(_ids(mt["task_ids"], "reference_date") for r in tasks["rounds"] for mt in r["model_tasks"]))


def target_end_dates(tasks: dict) -> set:
    return set().union(*(_ids(mt["task_ids"], "target_end_date") for r in tasks["rounds"] for mt in r["model_tasks"]))


def window(tasks: dict, ref: date) -> tuple[date, date]:
    due = tasks["rounds"][0].get("submissions_due", {"start": -6, "end": -3})
    return ref + timedelta(days=due["start"]), ref + timedelta(days=due["end"])


def _us_eastern_offset(utc: datetime) -> timedelta:
    """UTC-4 from the second Sunday of March 07:00 UTC to the first Sunday of November 06:00 UTC,
    else UTC-5 (US rules since 2007). Used when the system has no time-zone database."""
    y = utc.year
    mar = datetime(y, 3, 8) + timedelta(days=(6 - datetime(y, 3, 8).weekday()) % 7, hours=7)
    nov = datetime(y, 11, 1) + timedelta(days=(6 - datetime(y, 11, 1).weekday()) % 7, hours=6)
    return timedelta(hours=-4) if mar <= utc.replace(tzinfo=None) < nov else timedelta(hours=-5)


def now_eastern() -> datetime:
    try:
        return datetime.now(ZoneInfo("US/Eastern"))
    except Exception:  # no tz database installed
        utc = datetime.now(timezone.utc)
        return (utc + _us_eastern_offset(utc)).replace(tzinfo=None)
