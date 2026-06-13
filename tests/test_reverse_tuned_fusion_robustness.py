from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from evaluate_reverse_tuned_fusion_robustness import (  # noqa: E402
    load_robust_target,
    run_robustness,
)
from tests.test_reverse_fusion_source_holdout_tuning import (  # noqa: E402
    _toy_reverse_fixture,
    _write_predictions,
)


def _toy_robust_predictions(root: Path) -> Path:
    robust_root = root / "robust"
    target_paths = [root / f"robust_target_{index}.jpg" for index in range(6)]
    target_labels = [0, 0, 0, 1, 1, 1]
    _write_predictions(
        robust_root / "seed7" / "combined_v3" / "predictions.csv",
        target_labels,
        [0.25, 0.35, 0.45, 0.72, 0.82, 0.92],
        target_paths,
    )
    _write_predictions(
        robust_root / "seed7" / "resnet18" / "predictions.csv",
        target_labels,
        [0.15, 0.25, 0.35, 0.78, 0.72, 0.94],
        target_paths,
    )
    return robust_root


def _selected_configs(path: Path) -> None:
    pd.DataFrame(
        {
            "constraint_policy": ["cap_0p5"],
            "seed": [7],
            "fusion_c": [0.1],
            "dropout_config": ["none"],
            "source_fake_rate_cap": [0.5],
            "selection_validation_utility_mean": [1.0],
            "selection_validation_utility_min": [0.5],
        }
    ).to_csv(path, index=False)


def test_load_robust_target_aligns_prediction_scores(tmp_path: Path) -> None:
    robust_root = _toy_robust_predictions(tmp_path)

    target = load_robust_target(robust_root, 7, ["combined_v3", "resnet18"])

    assert list(target.columns) == ["path", "path_key", "y_true", "combined_v3", "resnet18"]
    assert len(target) == 6
    assert target["y_true"].tolist() == [0, 0, 0, 1, 1, 1]


def test_run_robustness_on_toy_fixture(tmp_path: Path) -> None:
    metadata = _toy_reverse_fixture(tmp_path)
    robust_root = _toy_robust_predictions(tmp_path)
    selected_path = tmp_path / "selected.csv"
    _selected_configs(selected_path)
    args = argparse.Namespace(
        run_root=str(tmp_path / "runs"),
        metadata=str(metadata),
        selected_configs=str(selected_path),
        robust_root=str(robust_root),
        variant="jpeg70",
        constraint_policy="cap_0p5",
        seeds=[7],
        methods=["combined_v3", "resnet18"],
        dropout_configs="none",
        threshold_tiebreak="higher",
        fake_detection_weight=1.0,
        real_clearance_weight=1.0,
        real_fpr_penalty=4.0,
        fake_miss_penalty=1.5,
        summary_dir=str(tmp_path / "assets"),
        report_path=str(tmp_path / "report.md"),
    )

    detail, summary = run_robustness(args)

    assert len(detail) == 1
    assert summary["variant"].iloc[0] == "jpeg70"
    assert summary["variant_policy"].iloc[0] == "cap_0p5"
    assert 0.0 <= detail["target_accuracy"].iloc[0] <= 1.0
