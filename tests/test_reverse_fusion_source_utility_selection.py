from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.select_reverse_fusion_by_source_utility import (  # noqa: E402
    concat_selection_frames,
    collect_candidate_metrics,
    select_candidates,
    source_decision_metrics,
    summarize_selection,
)


def _write_run(
    root: Path,
    candidate: str,
    seed: int,
    threshold: float,
    target_accuracy: float,
    target_predicted_positive_rate: float,
    source_scores: list[float],
) -> None:
    run_dir = root / "ms_cocoai_to_ishu_neural_fusion" / f"{candidate}_seed{seed}"
    (run_dir / "train").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "variant": ["train", "ishu_test"],
            "accuracy": [0.95, target_accuracy],
            "precision": [1.0, 0.70],
            "recall": [1.0, 0.80],
            "f1": [1.0, 0.75],
            "roc_auc": [0.99, 0.82],
            "brier_score": [0.05, 0.22],
            "expected_calibration_error": [0.10, 0.20],
            "maximum_calibration_error": [0.20, 0.30],
            "threshold": [threshold, threshold],
            "threshold_strategy": ["source_utility", "source_utility"],
            "threshold_source": ["source_train", "source_train"],
            "predicted_positive_rate": [0.50, target_predicted_positive_rate],
            "score_mean": [0.50, 0.60],
            "raw_score_mean": [0.50, 0.60],
            "n_samples": [4, 4],
        }
    ).to_csv(run_dir / "summary.csv", index=False)
    pd.DataFrame(
        {
            "path": [f"image_{index}.jpg" for index in range(4)],
            "y_true": [0, 0, 1, 1],
            "fake_score": source_scores,
        }
    ).to_csv(run_dir / "train" / "predictions.csv", index=False)


def test_source_decision_metrics_penalizes_real_false_positives(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions.csv"
    pd.DataFrame(
        {
            "y_true": [0, 0, 1, 1],
            "fake_score": [0.1, 0.8, 0.4, 0.9],
        }
    ).to_csv(predictions, index=False)

    metrics = source_decision_metrics(predictions, threshold=0.5)

    assert metrics["source_recall"] == 0.5
    assert metrics["source_specificity"] == 0.5
    assert metrics["source_real_fpr"] == 0.5
    assert metrics["source_predicted_positive_rate"] == 0.5
    assert metrics["source_utility"] == pytest.approx(-1.75)


def test_model_selection_can_apply_source_fake_rate_cap(tmp_path: Path) -> None:
    _write_run(
        tmp_path,
        "high_source_utility",
        seed=7,
        threshold=0.5,
        target_accuracy=0.60,
        target_predicted_positive_rate=0.90,
        source_scores=[0.1, 0.2, 0.8, 0.9],
    )
    _write_run(
        tmp_path,
        "capped_candidate",
        seed=7,
        threshold=0.75,
        target_accuracy=0.80,
        target_predicted_positive_rate=0.50,
        source_scores=[0.1, 0.2, 0.4, 0.9],
    )

    metrics = collect_candidate_metrics(
        tmp_path,
        "ms_cocoai_to_ishu_neural_fusion",
        seeds=[7],
        candidates=["high_source_utility", "capped_candidate"],
        variant="ishu_test",
    )
    unconstrained = select_candidates(metrics, cap=None)
    capped = select_candidates(metrics, cap=0.4)
    summary = summarize_selection(concat_selection_frames([unconstrained, capped]))

    assert unconstrained.iloc[0]["candidate"] == "high_source_utility"
    assert capped.iloc[0]["candidate"] == "capped_candidate"
    capped_summary = summary[summary["selection_policy"] == "source_utility_cap_0p4"].iloc[0]
    assert capped_summary["target_accuracy_mean"] == 0.80
    assert capped_summary["target_predicted_positive_rate_mean"] == 0.50
