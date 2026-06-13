from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from sweep_ishu_to_ms_threshold_objectives import (  # noqa: E402
    policy_label,
    run_sweep,
    summarize,
)


def _write_predictions(path: Path, scores: list[float], labels: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "path": [f"image_{index}.jpg" for index in range(len(scores))],
            "y_true": labels,
            "fake_score": scores,
        }
    ).to_csv(path, index=False)


def _toy_saved_fusion_fixture(root: Path) -> None:
    run = root / "runs" / "toy_family" / "ishu_seed7_to_ms_cocoai_all6"
    _write_predictions(
        run / "train" / "predictions.csv",
        scores=[0.05, 0.20, 0.65, 0.90],
        labels=[0, 0, 1, 1],
    )
    _write_predictions(
        run / "ms_cocoai" / "predictions.csv",
        scores=[0.10, 0.45, 0.55, 0.95],
        labels=[0, 0, 1, 1],
    )


def test_policy_label_includes_cap_and_weights() -> None:
    assert policy_label(None, 1.0, 1.0, 4.0, 1.5) == "uncapped_fd1_rc1_rfp4_fmp1p5"
    assert policy_label(0.45, 0.5, 1.0, 0.5, 3.0) == "cap0p45_fd0p5_rc1_rfp0p5_fmp3"


def test_summarize_sorts_by_target_accuracy() -> None:
    detail = pd.DataFrame(
        {
            "policy": ["low", "high"],
            "seed": [7, 7],
            "source_fake_rate_cap": [None, None],
            "fake_detection_weight": [1.0, 1.0],
            "real_clearance_weight": [1.0, 1.0],
            "real_fpr_penalty": [1.0, 1.0],
            "fake_miss_penalty": [1.0, 1.0],
            "target_accuracy": [0.5, 0.75],
            "target_roc_auc": [0.6, 0.8],
            "target_brier_score": [0.3, 0.2],
            "target_expected_calibration_error": [0.2, 0.1],
            "target_precision": [0.5, 0.8],
            "target_recall": [0.5, 0.7],
            "target_f1": [0.5, 0.75],
            "target_predicted_positive_rate": [0.4, 0.5],
            "source_accuracy": [0.5, 0.75],
            "source_predicted_positive_rate": [0.4, 0.5],
            "threshold": [0.5, 0.4],
            "threshold_source_utility": [0.0, 1.0],
        }
    )

    summary = summarize(detail)

    assert summary.iloc[0]["policy"] == "high"
    assert summary.iloc[0]["target_accuracy_mean"] == 0.75


def test_run_sweep_on_toy_saved_fusion_fixture(tmp_path: Path) -> None:
    _toy_saved_fusion_fixture(tmp_path)
    args = argparse.Namespace(
        run_root=str(tmp_path / "runs"),
        score_family="toy_family",
        run_template="ishu_seed{seed}_to_ms_cocoai_all6",
        target_variant="ms_cocoai",
        score_column="fake_score",
        seeds=[7],
        source_fake_rate_caps="none,0.5",
        fake_detection_weights="1",
        real_clearance_weights="1",
        real_fpr_penalties="1",
        fake_miss_penalties="1",
        threshold_tiebreak="higher",
        summary_dir=str(tmp_path / "assets"),
        report_path=str(tmp_path / "report.md"),
    )

    detail, summary = run_sweep(args)

    assert {"fixed_0p5", "uncapped_fd1_rc1_rfp1_fmp1", "cap0p5_fd1_rc1_rfp1_fmp1"} <= set(
        detail["policy"]
    )
    assert not summary.empty
    assert summary["n_seeds"].max() == 1
