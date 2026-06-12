from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_source_holdout_triage_tuning_selects_and_summarizes_operating_points(tmp_path: Path) -> None:
    paths = [tmp_path / f"image_{index}.jpg" for index in range(12)]
    labels = [0] * 6 + [1] * 3 + [1] * 3
    source_labels = [0] * 6 + [1] * 3 + [2] * 3
    scores = [0.02, 0.04, 0.08, 0.12, 0.18, 0.22, 0.72, 0.78, 0.83, 0.68, 0.74, 0.86]
    metadata_rows = []
    prediction_rows = []
    for path, label, source_label, score in zip(paths, labels, source_labels, scores, strict=True):
        metadata_rows.append(
            {
                "path": str(path),
                "split": "validation",
                "label": label,
                "source_label": source_label,
            }
        )
        prediction_rows.append({"path": str(path), "y_true": label, "fake_score": score})

    metadata_path = tmp_path / "metadata.csv"
    predictions_path = tmp_path / "predictions.csv"
    out_dir = tmp_path / "source_holdout_triage_tuning"
    pd.DataFrame(metadata_rows).to_csv(metadata_path, index=False)
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "summarize_source_holdout_triage_tuning.py"),
            "--metadata",
            str(metadata_path),
            "--predictions",
            f"seed3:toy={predictions_path}",
            "--out-dir",
            str(out_dir),
            "--split",
            "validation",
            "--seed",
            "3",
            "--max-real-fpr-grid",
            "0.0,0.2",
            "--max-fake-clearance-grid",
            "0.0,0.2",
            "--score-mode",
            "raw",
        ],
        cwd=ROOT,
        check=True,
    )

    grid = pd.read_csv(out_dir / "source_holdout_triage_grid.csv")
    tuned = pd.read_csv(out_dir / "source_holdout_triage_tuned.csv")
    summary = pd.read_csv(out_dir / "source_holdout_triage_tuned_summary.csv")
    assert len(grid) == 8
    assert set(tuned["group"]) == {"seed3"}
    assert set(tuned["method"]) == {"toy"}
    assert set(tuned["score_mode"]) == {"raw"}
    assert set(tuned["heldout_source"]) == {"sd21", "sdxl"}
    assert "calibration_utility" in tuned.columns
    assert "test_utility" in tuned.columns
    assert 0.0 <= summary.loc[0, "mean_test_coverage"] <= 1.0
    assert 0.0 <= summary.loc[0, "mean_test_triage_accuracy"] <= 1.0
    assert "mean_test_utility_ci_low" in summary.columns
    assert "selected_score_modes" in summary.columns
    assert (out_dir / "report.md").exists()
