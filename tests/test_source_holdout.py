from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_source_holdout_summary_uses_nonheldout_sources_for_threshold(tmp_path: Path) -> None:
    paths = [tmp_path / f"image_{index}.jpg" for index in range(12)]
    metadata_rows = []
    prediction_rows = []
    labels = [0] * 6 + [1] * 3 + [1] * 3
    source_labels = [0] * 6 + [1] * 3 + [2] * 3
    scores = [0.10, 0.12, 0.18, 0.20, 0.24, 0.28, 0.72, 0.76, 0.81, 0.62, 0.66, 0.70]
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
    out_dir = tmp_path / "source_holdout"
    pd.DataFrame(metadata_rows).to_csv(metadata_path, index=False)
    pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "summarize_source_holdout.py"),
            "--metadata",
            str(metadata_path),
            "--predictions",
            f"toy={predictions_path}",
            "--out-dir",
            str(out_dir),
            "--split",
            "validation",
            "--seed",
            "3",
        ],
        cwd=ROOT,
        check=True,
    )

    detail = pd.read_csv(out_dir / "source_holdout.csv")
    summary = pd.read_csv(out_dir / "source_holdout_method_summary.csv")
    assert set(detail["heldout_source"]) == {"sd21", "sdxl"}
    assert set(detail["method"]) == {"toy"}
    assert (detail["source_threshold_accuracy"] == 1.0).all()
    assert summary.loc[0, "method"] == "toy"
    assert summary.loc[0, "mean_source_threshold_accuracy"] == 1.0
    assert summary.loc[0, "n_holdouts"] == 2
    assert "mean_source_threshold_accuracy_ci_low" in summary.columns
    assert "mean_source_threshold_accuracy_ci_high" in summary.columns
    assert (out_dir / "report.md").exists()
