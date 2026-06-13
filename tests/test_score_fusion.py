from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _predictions(path: Path, scores: list[float], labels: list[int]) -> None:
    frame = pd.DataFrame(
        {
            "path": [str(path.parent / f"image-{index}.jpg") for index in range(len(scores))],
            "y_true": labels,
            "fake_score": scores,
        }
    )
    frame.to_csv(path, index=False)


def test_score_fusion_writes_train_and_variant_outputs(tmp_path: Path) -> None:
    labels = [0, 0, 1, 1, 0, 1]
    _predictions(tmp_path / "train_a.csv", [0.1, 0.3, 0.6, 0.8, 0.2, 0.7], labels)
    _predictions(tmp_path / "train_b.csv", [0.2, 0.1, 0.7, 0.6, 0.3, 0.8], labels)
    _predictions(tmp_path / "target_a.csv", [0.2, 0.4, 0.7, 0.9, 0.1, 0.6], labels)
    _predictions(tmp_path / "target_b.csv", [0.1, 0.2, 0.8, 0.7, 0.2, 0.9], labels)
    out_dir = tmp_path / "fusion"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "fuse_prediction_scores.py"),
            "--out-dir",
            str(out_dir),
            "--train",
            f"a={tmp_path / 'train_a.csv'}",
            "--train",
            f"b={tmp_path / 'train_b.csv'}",
            "--variant",
            f"target:a={tmp_path / 'target_a.csv'}",
            "--variant",
            f"target:b={tmp_path / 'target_b.csv'}",
            "--seed",
            "3",
            "--fusion-c",
            "0.5",
            "--branch-dropout-rate",
            "0.4",
            "--branch-dropout-repeats",
            "2",
        ],
        cwd=ROOT,
        check=True,
    )

    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(out_dir / "summary.csv")
    assert metrics["base_methods"] == ["a", "b"]
    assert metrics["n_fit"] == 18
    assert metrics["fusion_c"] == 0.5
    assert metrics["branch_dropout_rate"] == 0.4
    assert metrics["score_calibrator"] == "none"
    assert metrics["threshold_strategy"] == "fixed"
    assert metrics["threshold_source"] == "fixed"
    assert [row["variant"] for row in metrics["metrics"]] == ["train", "target"]
    assert {row["method"] for row in metrics["coefficients"]} == {"a", "b", "__intercept__"}
    train_predictions = pd.read_csv(out_dir / "train" / "predictions.csv")
    assert "raw_fake_score" in train_predictions.columns
    assert set(summary["score_calibrator"]) == {"none"}
    assert "brier_score" in summary.columns
    assert "expected_calibration_error" in summary.columns
    assert "predicted_positive_rate" in summary.columns
    assert set(summary["threshold_strategy"]) == {"fixed"}
    assert (out_dir / "train" / "predictions.csv").exists()
    assert (out_dir / "target" / "predictions.csv").exists()
    assert (out_dir / "score_fusion_coefficients.csv").exists()
    assert (out_dir / "score_fusion_model.joblib").exists()


def test_score_fusion_can_fit_heldout_score_calibrator(tmp_path: Path) -> None:
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    _predictions(tmp_path / "train_a.csv", [0.05, 0.20, 0.25, 0.35, 0.55, 0.70, 0.80, 0.90], labels)
    _predictions(tmp_path / "train_b.csv", [0.15, 0.10, 0.30, 0.40, 0.60, 0.65, 0.85, 0.95], labels)
    _predictions(tmp_path / "target_a.csv", [0.10, 0.25, 0.45, 0.50, 0.55, 0.65, 0.75, 0.88], labels)
    _predictions(tmp_path / "target_b.csv", [0.05, 0.30, 0.40, 0.45, 0.62, 0.70, 0.78, 0.92], labels)
    out_dir = tmp_path / "calibrated_fusion"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "fuse_prediction_scores.py"),
            "--out-dir",
            str(out_dir),
            "--train",
            f"a={tmp_path / 'train_a.csv'}",
            "--train",
            f"b={tmp_path / 'train_b.csv'}",
            "--variant",
            f"target:a={tmp_path / 'target_a.csv'}",
            "--variant",
            f"target:b={tmp_path / 'target_b.csv'}",
            "--seed",
            "5",
            "--score-calibrator",
            "temperature_balanced",
            "--calibration-fraction",
            "0.5",
        ],
        cwd=ROOT,
        check=True,
    )

    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(out_dir / "target" / "predictions.csv")
    summary = pd.read_csv(out_dir / "summary.csv")
    assert metrics["score_calibrator"] == "temperature_balanced"
    assert metrics["n_train"] == 8
    assert metrics["n_fusion_train"] == 4
    assert metrics["n_calibration"] == 4
    assert metrics["calibrator_temperature"] is not None
    assert set(summary["score_calibrator"]) == {"temperature_balanced"}
    assert "raw_fake_score" in predictions.columns
    assert (out_dir / "score_calibrator.joblib").exists()


def test_score_fusion_can_select_source_operating_threshold(tmp_path: Path) -> None:
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    _predictions(tmp_path / "train_a.csv", [0.05, 0.10, 0.20, 0.30, 0.60, 0.70, 0.80, 0.90], labels)
    _predictions(tmp_path / "train_b.csv", [0.10, 0.20, 0.25, 0.35, 0.55, 0.65, 0.75, 0.85], labels)
    _predictions(tmp_path / "target_a.csv", [0.05, 0.25, 0.35, 0.45, 0.55, 0.65, 0.78, 0.88], labels)
    _predictions(tmp_path / "target_b.csv", [0.08, 0.18, 0.30, 0.40, 0.58, 0.68, 0.76, 0.86], labels)
    out_dir = tmp_path / "thresholded_fusion"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "fuse_prediction_scores.py"),
            "--out-dir",
            str(out_dir),
            "--train",
            f"a={tmp_path / 'train_a.csv'}",
            "--train",
            f"b={tmp_path / 'train_b.csv'}",
            "--variant",
            f"target:a={tmp_path / 'target_a.csv'}",
            "--variant",
            f"target:b={tmp_path / 'target_b.csv'}",
            "--seed",
            "11",
            "--fusion-c",
            "0.1",
            "--threshold",
            "0.99",
            "--threshold-strategy",
            "source_accuracy",
            "--calibration-fraction",
            "0.5",
        ],
        cwd=ROOT,
        check=True,
    )

    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(out_dir / "summary.csv")
    assert metrics["requested_threshold"] == 0.99
    assert metrics["threshold"] < 0.99
    assert metrics["threshold_strategy"] == "source_accuracy"
    assert metrics["threshold_source"] == "source_calibration"
    assert metrics["n_fusion_train"] == 4
    assert metrics["n_calibration"] == 4
    assert metrics["threshold_selection_utility"] >= 0.5
    assert set(summary["threshold_strategy"]) == {"source_accuracy"}
    assert set(summary["threshold_source"]) == {"source_calibration"}
