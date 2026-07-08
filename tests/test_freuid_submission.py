from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _write_sample(path: Path) -> None:
    pd.DataFrame({"id": ["a", "b", "c"], "label": [0, 0, 0]}).to_csv(path, index=False)


def test_freuid_packager_emits_constant_baseline_in_sample_order(tmp_path: Path) -> None:
    sample = tmp_path / "sample_submission.csv"
    submission = tmp_path / "submission.csv"
    manifest_path = tmp_path / "submission_manifest.json"
    _write_sample(sample)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "package_freuid_submission.py"),
            "--sample-submission",
            str(sample),
            "--out-path",
            str(submission),
            "--manifest-out",
            str(manifest_path),
        ],
        cwd=ROOT,
        check=True,
    )

    frame = pd.read_csv(submission)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert list(frame.columns) == ["id", "label"]
    assert list(frame["id"]) == ["a", "b", "c"]
    assert list(frame["label"]) == [0, 0, 0]
    assert manifest["label_counts"] == {"0": 3}
    assert manifest["matched_predictions"] == 0


def test_freuid_packager_accepts_path_predictions_and_threshold(tmp_path: Path) -> None:
    sample = tmp_path / "sample_submission.csv"
    predictions = tmp_path / "predictions.csv"
    submission = tmp_path / "submission.csv"
    _write_sample(sample)
    pd.DataFrame(
        {
            "image_path": ["public_test/b.jpeg", "public_test/a.jpeg", "public_test/c.jpeg"],
            "fraud_score": [0.51, 0.49, 0.90],
        }
    ).to_csv(predictions, index=False)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "package_freuid_submission.py"),
            "--sample-submission",
            str(sample),
            "--predictions",
            str(predictions),
            "--out-path",
            str(submission),
            "--threshold",
            "0.5",
        ],
        cwd=ROOT,
        check=True,
    )

    frame = pd.read_csv(submission)
    assert list(frame["id"]) == ["a", "b", "c"]
    assert list(frame["label"]) == [0, 1, 1]


def test_freuid_linter_passes_good_submission_and_reports_counts(tmp_path: Path) -> None:
    sample = tmp_path / "sample_submission.csv"
    submission = tmp_path / "submission.csv"
    manifest_path = tmp_path / "lint.json"
    _write_sample(sample)
    pd.DataFrame({"id": ["a", "b", "c"], "label": [0, 1, 1]}).to_csv(submission, index=False)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "lint_freuid_submission.py"),
            "--sample-submission",
            str(sample),
            "--submission",
            str(submission),
            "--manifest-out",
            str(manifest_path),
        ],
        cwd=ROOT,
        check=True,
    )

    report = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = {row["check"]: row for row in report["checks"]}
    assert report["status"] == "pass"
    assert report["label_counts"] == {"0": 1, "1": 2}
    assert checks["submission_columns"]["passed"] is True
    assert checks["sample_id_set"]["passed"] is True


def test_freuid_linter_fails_extra_column_and_bad_ids(tmp_path: Path) -> None:
    sample = tmp_path / "sample_submission.csv"
    submission = tmp_path / "submission.csv"
    manifest_path = tmp_path / "lint.json"
    _write_sample(sample)
    pd.DataFrame({"id": ["a", "b", "x"], "label": [0, 2, 1], "score": [0.1, 0.2, 0.3]}).to_csv(
        submission,
        index=False,
    )

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "lint_freuid_submission.py"),
            "--sample-submission",
            str(sample),
            "--submission",
            str(submission),
            "--manifest-out",
            str(manifest_path),
        ],
        cwd=ROOT,
        check=False,
    )

    report = json.loads(manifest_path.read_text(encoding="utf-8"))
    checks = {row["check"]: row for row in report["checks"]}
    assert result.returncode == 1
    assert report["status"] == "fail"
    assert checks["submission_columns"]["passed"] is False
