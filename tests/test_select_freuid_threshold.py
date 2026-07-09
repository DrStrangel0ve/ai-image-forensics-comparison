from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_select_freuid_threshold_writes_manifest_and_thresholded_predictions(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.csv"
    manifest_path = tmp_path / "threshold.json"
    thresholded = tmp_path / "thresholded.csv"
    pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e", "f"],
            "y_true": [0, 0, 0, 1, 1, 1],
            "fraud_score": [0.05, 0.15, 0.25, 0.45, 0.70, 0.90],
        }
    ).to_csv(predictions, index=False)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "select_freuid_threshold.py"),
            "--predictions",
            str(predictions),
            "--out-json",
            str(manifest_path),
            "--out-predictions",
            str(thresholded),
            "--bpcer-target",
            "0.01",
        ],
        cwd=ROOT,
        check=True,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(thresholded)
    assert manifest["n_rows"] == 6
    assert manifest["id_column"] == "id"
    assert manifest["label_column"] == "y_true"
    assert manifest["score_column"] == "fraud_score"
    assert manifest["bpcer_at_operating_point"] <= 0.01
    assert set(manifest["label_counts"]) <= {"0", "1"}
    assert list(frame.columns) == ["id", "y_true", "fraud_score", "label"]
    assert list(frame["id"]) == ["a", "b", "c", "d", "e", "f"]
    assert set(frame["label"].unique()).issubset({0, 1})
