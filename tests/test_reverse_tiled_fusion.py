from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from evaluate_reverse_tiled_fusion import (  # noqa: E402
    load_tiled_target,
    run_tiled_fusion,
    summarize_by_score_mode,
)
from tests.test_reverse_fusion_source_holdout_tuning import (  # noqa: E402
    _toy_reverse_fixture,
)


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


def _tile_detail(path: Path, root: Path, use_matching_paths: bool = False) -> None:
    paths = (
        [root / f"target_{index}.jpg" for index in range(6)]
        if use_matching_paths
        else [root / "copied_target" / f"{index:06d}.png" for index in range(6)]
    )
    pd.DataFrame(
        {
            "seed": [7] * 6,
            "path": [str(value) for value in paths],
            "y_true": [0, 0, 0, 1, 1, 1],
            "global_score": [0.2, 0.3, 0.4, 0.7, 0.8, 0.9],
            "tile_mean_score": [0.25, 0.32, 0.42, 0.68, 0.76, 0.88],
            "tile_max_score": [0.35, 0.5, 0.7, 0.9, 0.95, 0.98],
            "tile_top2_mean_score": [0.3, 0.42, 0.55, 0.82, 0.9, 0.95],
        }
    ).to_csv(path, index=False)


def _clean_target(root: Path) -> pd.DataFrame:
    paths = [root / f"target_{index}.jpg" for index in range(6)]
    return pd.DataFrame(
        {
            "path": [str(value) for value in paths],
            "path_key": [str(value.resolve()).replace("\\", "/").lower() for value in paths],
            "y_true": [0, 0, 0, 1, 1, 1],
            "combined_v3": [0.2, 0.3, 0.4, 0.7, 0.8, 0.9],
            "resnet18": [0.1, 0.2, 0.3, 0.75, 0.7, 0.95],
        }
    )


def test_load_tiled_target_uses_path_alignment_when_available(tmp_path: Path) -> None:
    detail_path = tmp_path / "tile_detail.csv"
    _tile_detail(detail_path, tmp_path, use_matching_paths=True)

    target = load_tiled_target(
        _clean_target(tmp_path),
        pd.read_csv(detail_path),
        seed=7,
        branch="combined_v3",
        score_mode="tile_top2_mean",
    )

    assert target["combined_v3"].tolist() == [0.3, 0.42, 0.55, 0.82, 0.9, 0.95]


def test_load_tiled_target_falls_back_to_order_verified_scores(tmp_path: Path) -> None:
    detail_path = tmp_path / "tile_detail.csv"
    _tile_detail(detail_path, tmp_path, use_matching_paths=False)

    target = load_tiled_target(
        _clean_target(tmp_path),
        pd.read_csv(detail_path),
        seed=7,
        branch="combined_v3",
        score_mode="tile_max",
    )

    assert target["combined_v3"].tolist() == [0.35, 0.5, 0.7, 0.9, 0.95, 0.98]


def test_run_tiled_fusion_on_toy_fixture(tmp_path: Path) -> None:
    metadata = _toy_reverse_fixture(tmp_path)
    selected_path = tmp_path / "selected.csv"
    detail_path = tmp_path / "tile_detail.csv"
    _selected_configs(selected_path)
    _tile_detail(detail_path, tmp_path)
    args = argparse.Namespace(
        run_root=str(tmp_path / "runs"),
        metadata=str(metadata),
        selected_configs=str(selected_path),
        tile_detail=str(detail_path),
        tile_branch="combined_v3",
        tile_score_modes=["global", "tile_top2_mean"],
        constraint_policy="cap_0p5",
        seeds=[7],
        methods=["combined_v3", "resnet18"],
        dropout_configs="none",
        threshold_tiebreak="higher",
        fake_detection_weight=1.0,
        real_clearance_weight=1.0,
        real_fpr_penalty=4.0,
        fake_miss_penalty=1.5,
        alignment_tolerance=1e-5,
        summary_dir=str(tmp_path / "assets"),
        report_path=str(tmp_path / "report.md"),
    )

    detail, summary = run_tiled_fusion(args)

    assert set(detail["score_mode"]) == {"global", "tile_top2_mean"}
    assert set(summary["score_mode"]) == {"global", "tile_top2_mean"}
    assert 0.0 <= detail["target_accuracy"].iloc[0] <= 1.0


def test_summarize_by_score_mode_keeps_modes_separate() -> None:
    detail = pd.DataFrame(
        {
            "score_mode": ["global", "global", "tile_max", "tile_max"],
            "constraint_policy": ["cap", "cap", "cap", "cap"],
            "seed": [1, 2, 1, 2],
            "target_accuracy": [0.5, 0.7, 0.6, 0.8],
            "target_roc_auc": [0.6, 0.8, 0.7, 0.9],
            "target_brier_score": [0.2, 0.3, 0.2, 0.3],
            "target_expected_calibration_error": [0.1, 0.2, 0.1, 0.2],
            "target_precision": [0.5, 0.7, 0.6, 0.8],
            "target_recall": [0.5, 0.7, 0.6, 0.8],
            "target_f1": [0.5, 0.7, 0.6, 0.8],
            "target_predicted_positive_rate": [0.4, 0.6, 0.5, 0.7],
            "source_predicted_positive_rate": [0.4, 0.4, 0.4, 0.4],
            "threshold_source_predicted_positive_rate": [0.4, 0.4, 0.4, 0.4],
            "fusion_c": [0.1, 0.1, 0.1, 0.1],
            "dropout_config": ["none", "none", "none", "none"],
            "source_fake_rate_cap": [0.4, 0.4, 0.4, 0.4],
        }
    )

    summary = summarize_by_score_mode(detail)

    assert summary.set_index("score_mode").loc["global", "target_accuracy_mean"] == 0.6
    assert summary.set_index("score_mode").loc["tile_max", "target_accuracy_mean"] == 0.7
