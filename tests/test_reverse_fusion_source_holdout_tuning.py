from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from tune_reverse_fusion_source_holdout import (  # noqa: E402
    DropoutConfig,
    _parse_dropout_configs,
    evaluate_source_holdout_grid,
    evaluate_final_config,
    load_seed_frames,
    run_tuning,
    select_config,
    summarize_grid,
)


def _write_predictions(path: Path, labels: list[int], scores: list[float], paths: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "path": [str(value) for value in paths],
            "y_true": labels,
            "fake_score": scores,
        }
    ).to_csv(path, index=False)


def _write_branch_run(
    root: Path,
    family: str,
    run_name: str,
    train_labels: list[int],
    train_scores: list[float],
    train_paths: list[Path],
    target_labels: list[int],
    target_scores: list[float],
    target_paths: list[Path],
) -> None:
    _write_predictions(
        root / "runs" / family / run_name / "predictions.csv",
        train_labels,
        train_scores,
        train_paths,
    )
    _write_predictions(
        root / "runs" / family / f"{run_name}_to_ishu_test" / "predictions.csv",
        target_labels,
        target_scores,
        target_paths,
    )


def _toy_reverse_fixture(root: Path) -> Path:
    train_paths = [root / f"source_{index}.jpg" for index in range(8)]
    target_paths = [root / f"target_{index}.jpg" for index in range(6)]
    train_labels = [0, 0, 0, 0, 1, 1, 1, 1]
    target_labels = [0, 0, 0, 1, 1, 1]
    metadata = root / "metadata.csv"
    pd.DataFrame(
        {
            "path": [str(path) for path in train_paths],
            "label": train_labels,
            "source_label": [0, 0, 0, 0, 1, 1, 2, 2],
        }
    ).to_csv(metadata, index=False)
    _write_branch_run(
        root,
        "ms_cocoai_to_ishu_neural_fusion",
        "combined_v3_seed7",
        train_labels,
        [0.1, 0.2, 0.3, 0.4, 0.7, 0.75, 0.7, 0.8],
        train_paths,
        target_labels,
        [0.2, 0.3, 0.4, 0.7, 0.8, 0.9],
        target_paths,
    )
    _write_branch_run(
        root,
        "ms_cocoai_to_ishu_neural_fusion",
        "resnet18_seed7",
        train_labels,
        [0.2, 0.1, 0.4, 0.3, 0.65, 0.85, 0.6, 0.9],
        train_paths,
        target_labels,
        [0.1, 0.2, 0.3, 0.75, 0.7, 0.95],
        target_paths,
    )
    return metadata


def test_parse_dropout_configs() -> None:
    configs = _parse_dropout_configs("none,mean0p35x8,neutral0p25x4")

    assert configs[0] == DropoutConfig(label="none")
    assert configs[1].fill == "mean"
    assert configs[1].rate == 0.35
    assert configs[1].repeats == 8
    assert configs[2].fill == "neutral"


def test_source_holdout_tuning_grid_selects_and_evaluates(tmp_path: Path) -> None:
    metadata = _toy_reverse_fixture(tmp_path)
    source, target, methods = load_seed_frames(
        tmp_path / "runs",
        metadata,
        ["combined_v3", "resnet18"],
        seed=7,
    )

    folds = evaluate_source_holdout_grid(
        source,
        methods,
        fusion_cs=[0.1, 1.0],
        dropouts=[DropoutConfig(label="none")],
        caps=[None, 0.5],
        seed=7,
    )
    summary = summarize_grid(folds)
    selected_config = select_config(summary, "mean")
    selected = evaluate_final_config(
        source,
        target,
        methods,
        selected_config,
        {"none": DropoutConfig(label="none")},
        seed=7,
    )

    assert set(folds["heldout_source_label"]) == {1, 2}
    assert selected["target_accuracy"] >= 0.5
    assert 0.0 <= selected["target_predicted_positive_rate"] <= 1.0


def test_run_tuning_writes_expected_frames(tmp_path: Path) -> None:
    metadata = _toy_reverse_fixture(tmp_path)
    args = argparse.Namespace(
        run_root=str(tmp_path / "runs"),
        metadata=str(metadata),
        summary_dir=str(tmp_path / "assets"),
        report_path=str(tmp_path / "report.md"),
        seeds=[7],
        methods=["combined_v3", "resnet18"],
        fusion_cs="0.1",
        dropout_configs="none",
        source_fake_rate_caps="0.5",
        selection_score="mean",
        real_validation_fraction=0.5,
        fake_detection_weight=1.0,
        real_clearance_weight=1.0,
        real_fpr_penalty=4.0,
        fake_miss_penalty=1.5,
        threshold_tiebreak="higher",
    )

    folds, grid, selected, final = run_tuning(args)

    assert len(folds) == 2
    assert len(grid) == 1
    assert len(selected) == 1
    assert final["selection_policy"].iloc[0] == "source_holdout_tuned_fusion"
