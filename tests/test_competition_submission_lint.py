from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_competition_submission_lint_passes_packager_output_with_expected_ids(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.csv"
    submission = tmp_path / "submission.csv"
    package_manifest = tmp_path / "submission_manifest.json"
    lint_manifest = tmp_path / "submission_lint.json"
    expected_ids = tmp_path / "expected_ids.csv"

    pd.DataFrame(
        {
            "path": [
                str(tmp_path / "images" / "img-b.jpg"),
                str(tmp_path / "images" / "img-a.jpg"),
                str(tmp_path / "images" / "img-c.jpg"),
            ],
            "y_true": [0, 0, 1],
            "fake_score": [0.45, 0.10, 0.92],
        }
    ).to_csv(predictions, index=False)
    pd.DataFrame({"path": ["images/img-a.jpg", "images/img-b.jpg", "images/img-c.jpg"]}).to_csv(
        expected_ids, index=False
    )

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "package_competition_submission.py"),
            "--predictions",
            str(predictions),
            "--out-path",
            str(submission),
            "--manifest-out",
            str(package_manifest),
            "--sort-by-id",
        ],
        cwd=ROOT,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "lint_competition_submission.py"),
            "--submission",
            str(submission),
            "--expected-ids",
            str(expected_ids),
            "--manifest-out",
            str(lint_manifest),
        ],
        cwd=ROOT,
        check=True,
    )

    report = json.loads(lint_manifest.read_text(encoding="utf-8"))
    checks = {row["check"]: row for row in report["checks"]}
    assert report["status"] == "pass"
    assert report["n_rows"] == 3
    assert checks["no_label_leakage_columns"]["passed"] is True
    assert checks["expected_id_coverage"]["passed"] is True
    assert checks["triage_decision_consistent"]["passed"] is True
    assert report["expected_id_summary"] == {
        "expected_count": 3,
        "missing_count": 0,
        "extra_count": 0,
        "missing_examples": [],
        "extra_examples": [],
    }


def test_competition_submission_lint_fails_label_leakage_and_bad_labels(tmp_path: Path) -> None:
    submission = tmp_path / "bad_submission.csv"
    lint_manifest = tmp_path / "bad_submission_lint.json"
    pd.DataFrame(
        {
            "image_id": ["a", "b"],
            "fake_score": [0.9, 0.1],
            "predicted_label": [0, 0],
            "y_true": [1, 0],
        }
    ).to_csv(submission, index=False)

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "lint_competition_submission.py"),
            "--submission",
            str(submission),
            "--manifest-out",
            str(lint_manifest),
        ],
        cwd=ROOT,
        check=False,
    )

    report = json.loads(lint_manifest.read_text(encoding="utf-8"))
    checks = {row["check"]: row for row in report["checks"]}
    assert result.returncode == 1
    assert report["status"] == "fail"
    assert checks["no_label_leakage_columns"]["passed"] is False
    assert checks["predicted_label_consistent"]["passed"] is False
