from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_group_rank_calibration_removes_group_score_offsets(tmp_path: Path) -> None:
    val_predictions = tmp_path / "val_predictions.csv"
    val_metadata = tmp_path / "val_metadata.csv"
    test_diagnostics = tmp_path / "test_diagnostics.csv"
    output_dir = tmp_path / "out"
    pd.DataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "y_true": [0, 1, 0, 1],
            "fraud_score": [0.1, 0.2, 0.8, 0.9],
        }
    ).to_csv(val_predictions, index=False)
    pd.DataFrame({"id": ["a", "b", "c", "d"], "type": ["x", "x", "y", "y"]}).to_csv(
        val_metadata, index=False
    )
    pd.DataFrame(
        {
            "id": ["u", "v", "w", "z"],
            "predicted_type_index": [0, 0, 1, 1],
            "fraud_score": [0.8, 0.9, 0.1, 0.2],
        }
    ).to_csv(test_diagnostics, index=False)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "calibrate_freuid_group_ranks.py"),
            "--val-predictions",
            str(val_predictions),
            "--val-metadata",
            str(val_metadata),
            "--test-diagnostics",
            str(test_diagnostics),
            "--output-dir",
            str(output_dir),
            "--global-weights",
            "0",
            "1",
        ],
        cwd=ROOT,
        check=True,
    )

    pure_group = pd.read_csv(output_dir / "test_predictions_global_0p00.csv")
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert pure_group["fraud_score"].tolist() == [0.5, 1.0, 0.5, 1.0]
    assert summary["n_val"] == 4
    assert summary["n_test"] == 4
