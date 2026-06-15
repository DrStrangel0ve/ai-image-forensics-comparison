from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_submission_deadline_calendar_builds_csv_markdown_and_ics(tmp_path: Path) -> None:
    critical_path = tmp_path / "critical_path.csv"
    pd.DataFrame(
        {
            "venue_key": ["DFRWS", "WIFS"],
            "venue": ["DFRWS-USA 2026 poster/demo", "IEEE WIFS 2026 paper"],
            "deadline": ["2026-07-07", "2026-07-15"],
            "packet_status": ["ready_to_polish", "ready_to_polish"],
            "phase": ["export", "draft"],
            "priority": [3, 2],
            "item": ["final upload export", "paper prose sections"],
            "status": ["final_export_needed", "writing_needed"],
            "target_done_by": ["2026-07-04", "2026-07-05"],
            "days_until_target": [19, 20],
            "urgency": ["scheduled", "scheduled"],
            "days_until_deadline": [22, 30],
            "reason": ["reason", "reason"],
            "action": ["Export the final poster.", "Edit the paper prose."],
            "paths": ["poster.pptx", "draft.md"],
        }
    ).to_csv(critical_path, index=False)
    out_path = tmp_path / "deadline_calendar.md"
    csv_out = tmp_path / "deadline_calendar.csv"
    ics_out = tmp_path / "deadline_calendar.ics"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_submission_deadline_calendar.py"),
            "--critical-path",
            str(critical_path),
            "--out-path",
            str(out_path),
            "--csv-out",
            str(csv_out),
            "--ics-out",
            str(ics_out),
            "--run-date",
            "2026-06-15",
        ],
        cwd=ROOT,
        check=True,
    )

    report = out_path.read_text(encoding="utf-8")
    events = pd.read_csv(csv_out)
    calendar = ics_out.read_text(encoding="utf-8")

    assert "Submission Deadline Calendar" in report
    assert len(events) == 4
    assert set(events["event_type"]) == {"target_done_by", "venue_deadline"}
    assert "DFRWS final submission deadline" in set(events["title"])
    assert "WIFS target: paper prose sections" in set(events["title"])
    assert "BEGIN:VCALENDAR" in calendar
    assert calendar.count("BEGIN:VEVENT") == 4
    assert "DTSTART;VALUE=DATE:20260704" in calendar
    assert "DTSTART;VALUE=DATE:20260715" in calendar
