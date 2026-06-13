from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_reverse_source_utility_sweep import (  # noqa: E402
    collect_metrics,
    config_key,
    summarize_metrics,
)


def _write_summary(path: Path, accuracy: float, predicted_positive_rate: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "variant": ["train", "ishu_test"],
            "score_calibrator": ["none", "none"],
            "accuracy": [0.95, accuracy],
            "precision": [0.90, 0.70],
            "recall": [0.92, 0.86],
            "f1": [0.91, 0.77],
            "roc_auc": [0.99, 0.83],
            "brier_score": [0.05, 0.22],
            "expected_calibration_error": [0.12, 0.21],
            "maximum_calibration_error": [0.30, 0.45],
            "threshold": [0.65, 0.65],
            "threshold_strategy": ["source_utility", "source_utility"],
            "threshold_tiebreak": ["higher", "higher"],
            "threshold_source": ["source_calibration", "source_calibration"],
            "threshold_fake_detection_weight": [1.0, 1.0],
            "threshold_real_clearance_weight": [1.0, 1.0],
            "threshold_real_fpr_penalty": [4.0, 4.0],
            "threshold_fake_miss_penalty": [1.5, 1.5],
            "threshold_max_positive_rate": [0.48, 0.48],
            "threshold_source_predicted_positive_rate": [0.475, 0.475],
            "predicted_positive_rate": [0.45, predicted_positive_rate],
            "score_mean": [0.50, 0.70],
            "raw_score_mean": [0.50, 0.70],
            "n_samples": [200, 114],
        }
    ).to_csv(path, index=False)


def test_source_utility_sweep_collects_and_summarizes_metrics(tmp_path: Path) -> None:
    key = config_key(real_fpr_penalty=4.0, fake_miss_penalty=1.5, cap=0.48)
    assert key.endswith("rfp4_fmp1p5_cap0p48")
    assert config_key(real_fpr_penalty=4.0, fake_miss_penalty=1.5, cap=None).endswith(
        "rfp4_fmp1p5_nocap"
    )
    configs = [(key, 4.0, 1.5, 0.48)]
    for seed, accuracy, predicted_positive_rate in [(7, 0.72, 0.62), (17, 0.66, 0.70)]:
        _write_summary(
            tmp_path
            / "ms_cocoai_to_ishu_neural_fusion"
            / f"{key}_seed{seed}"
            / "summary.csv",
            accuracy=accuracy,
            predicted_positive_rate=predicted_positive_rate,
        )

    metrics = collect_metrics(tmp_path, [7, 17], configs)
    summary = summarize_metrics(metrics)
    target = summary[summary["variant"] == "ishu_test"].iloc[0]

    assert set(metrics["threshold_strategy"]) == {"source_utility"}
    assert target["n_seeds"] == 2
    assert target["accuracy_mean"] == 0.69
    assert target["predicted_positive_rate_mean"] == pytest.approx(0.66)
    assert target["threshold_source_predicted_positive_rate_mean"] == 0.475
