from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_submission_critical_path_orders_pending_upload_work(tmp_path: Path) -> None:
    checklist = tmp_path / "submission_upload_checklist.csv"
    scorecard = tmp_path / "submission_scorecard.csv"
    out_path = tmp_path / "submission_critical_path.md"
    csv_out = tmp_path / "submission_critical_path.csv"

    pd.DataFrame(
        [
            {
                "venue_key": "DFRWS",
                "venue": "DFRWS-USA 2026 poster/demo",
                "deadline": "2026-07-07",
                "item": "poster abstract",
                "status": "ready_asset",
                "paths": "reports/submission_text_drafts_2026_06_14.md",
                "action": "Use the abstract.",
                "paths_present": True,
                "missing_paths": "",
            },
            {
                "venue_key": "DFRWS",
                "venue": "DFRWS-USA 2026 poster/demo",
                "deadline": "2026-07-07",
                "item": "final upload export",
                "status": "final_export_needed",
                "paths": "reports/assets/dfrws_poster_draft_v2_2026_06_13.pptx",
                "action": "Export the poster.",
                "paths_present": True,
                "missing_paths": "",
            },
            {
                "venue_key": "WIFS",
                "venue": "IEEE WIFS 2026 paper",
                "deadline": "2026-07-15",
                "item": "paper prose sections",
                "status": "writing_needed",
                "paths": "reports/wifs_manuscript_draft_2026_06_14.md",
                "action": "Edit the WIFS draft.",
                "paths_present": True,
                "missing_paths": "",
            },
        ]
    ).to_csv(checklist, index=False)

    pd.DataFrame(
        [
            {"venue_key": "DFRWS", "packet_status": "ready_to_polish"},
            {"venue_key": "WIFS", "packet_status": "ready_to_polish"},
        ]
    ).to_csv(scorecard, index=False)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_submission_critical_path.py"),
            "--checklist",
            str(checklist),
            "--scorecard",
            str(scorecard),
            "--out-path",
            str(out_path),
            "--csv-out",
            str(csv_out),
            "--run-date",
            "2026-06-14",
        ],
        cwd=ROOT,
        check=True,
    )

    text = out_path.read_text(encoding="utf-8")
    critical_path = pd.read_csv(csv_out)

    assert "Submission Critical Path" in text
    assert "ready_to_polish" in text
    assert set(critical_path["item"]) == {"final upload export", "paper prose sections"}
    assert "poster abstract" not in set(critical_path["item"])
    assert critical_path.iloc[0]["item"] == "final upload export"
    assert critical_path.iloc[0]["target_done_by"] == "2026-07-04"
    assert critical_path.iloc[0]["urgency"] == "scheduled"
    assert critical_path.iloc[1]["target_done_by"] == "2026-07-05"
    assert set(critical_path["urgency"]) == {"scheduled"}
    assert "writing_needed" in text
    assert "final_export_needed" in text
    assert "urgency" in text
    assert "due_next_two_weeks" in text


def test_submission_critical_path_handles_no_pending_work(tmp_path: Path) -> None:
    checklist = tmp_path / "submission_upload_checklist.csv"
    scorecard = tmp_path / "submission_scorecard.csv"
    out_path = tmp_path / "submission_critical_path.md"
    csv_out = tmp_path / "submission_critical_path.csv"

    pd.DataFrame(
        [
            {
                "venue_key": "DFRWS",
                "venue": "DFRWS-USA 2026 poster/demo",
                "deadline": "2026-07-07",
                "item": "poster abstract",
                "status": "ready_asset",
                "paths": "reports/submission_text_drafts_2026_06_14.md",
                "action": "Use the abstract.",
                "paths_present": True,
                "missing_paths": "",
            }
        ]
    ).to_csv(checklist, index=False)
    pd.DataFrame([{"venue_key": "DFRWS", "packet_status": "ready_to_polish"}]).to_csv(
        scorecard,
        index=False,
    )

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_submission_critical_path.py"),
            "--checklist",
            str(checklist),
            "--scorecard",
            str(scorecard),
            "--out-path",
            str(out_path),
            "--csv-out",
            str(csv_out),
            "--run-date",
            "2026-06-14",
        ],
        cwd=ROOT,
        check=True,
    )

    assert "Submission Critical Path" in out_path.read_text(encoding="utf-8")
    assert pd.read_csv(csv_out).empty


def test_submission_critical_path_marks_near_targets(tmp_path: Path) -> None:
    checklist = tmp_path / "submission_upload_checklist.csv"
    scorecard = tmp_path / "submission_scorecard.csv"
    out_path = tmp_path / "submission_critical_path.md"
    csv_out = tmp_path / "submission_critical_path.csv"

    pd.DataFrame(
        [
            {
                "venue_key": "DFRWS",
                "venue": "DFRWS-USA 2026 poster/demo",
                "deadline": "2026-07-07",
                "item": "final upload export",
                "status": "final_export_needed",
                "paths": "reports/assets/dfrws_poster_draft_v2_2026_06_13.pptx",
                "action": "Export the poster.",
                "paths_present": True,
                "missing_paths": "",
            }
        ]
    ).to_csv(checklist, index=False)
    pd.DataFrame([{"venue_key": "DFRWS", "packet_status": "ready_to_polish"}]).to_csv(
        scorecard,
        index=False,
    )

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_submission_critical_path.py"),
            "--checklist",
            str(checklist),
            "--scorecard",
            str(scorecard),
            "--out-path",
            str(out_path),
            "--csv-out",
            str(csv_out),
            "--run-date",
            "2026-06-27",
        ],
        cwd=ROOT,
        check=True,
    )

    critical_path = pd.read_csv(csv_out)
    assert critical_path.iloc[0]["days_until_target"] == 7
    assert critical_path.iloc[0]["urgency"] == "due_this_week"
