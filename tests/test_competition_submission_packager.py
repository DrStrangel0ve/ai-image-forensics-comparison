from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_competition_packager_emits_submission_without_labels(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.csv"
    out_path = tmp_path / "submission.csv"
    manifest_path = tmp_path / "submission_manifest.json"
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

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "package_competition_submission.py"),
            "--predictions",
            str(predictions),
            "--out-path",
            str(out_path),
            "--manifest-out",
            str(manifest_path),
            "--sort-by-id",
        ],
        cwd=ROOT,
        check=True,
    )

    submission = pd.read_csv(out_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert list(submission.columns) == [
        "image_id",
        "fake_score",
        "predicted_label",
        "predicted_label_name",
        "confidence",
        "triage_decision",
    ]
    assert list(submission["image_id"]) == ["img-a", "img-b", "img-c"]
    assert "y_true" not in submission.columns
    assert list(submission["predicted_label_name"]) == ["real", "real", "fake"]
    assert list(submission["triage_decision"]) == ["likely_real", "uncertain", "likely_fake"]
    assert submission.loc[submission["image_id"] == "img-c", "confidence"].iloc[0] == 0.84
    assert manifest["n_rows"] == 3
    assert manifest["id_column"] == "path"
    assert manifest["score_column"] == "fake_score"
    assert manifest["y_true_present_excluded"] is True
    assert manifest["triage_counts"] == {"likely_fake": 1, "likely_real": 1, "uncertain": 1}


def test_competition_packager_accepts_explicit_columns(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.csv"
    out_path = tmp_path / "submission.csv"
    pd.DataFrame(
        {
            "image_id": ["b", "a"],
            "prob_fake": [0.51, 0.49],
        }
    ).to_csv(predictions, index=False)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "package_competition_submission.py"),
            "--predictions",
            str(predictions),
            "--out-path",
            str(out_path),
            "--id-column",
            "image_id",
            "--score-column",
            "prob_fake",
            "--decision-threshold",
            "0.55",
        ],
        cwd=ROOT,
        check=True,
    )

    submission = pd.read_csv(out_path)
    assert list(submission["image_id"]) == ["b", "a"]
    assert list(submission["predicted_label"]) == [0, 0]
