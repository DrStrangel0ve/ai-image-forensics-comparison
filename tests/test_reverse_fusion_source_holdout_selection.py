from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

from select_reverse_fusion_by_source_holdout import (  # noqa: E402
    collect_source_holdout_metrics,
    load_source_metadata,
    select_candidates,
    source_decision_metrics_from_frame,
    summarize_policy_source_holdouts,
    summarize_selection,
)
from select_reverse_fusion_by_source_utility import concat_selection_frames  # noqa: E402


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
            "n_samples": [6, 4],
        }
    ).to_csv(run_dir / "summary.csv", index=False)
    pd.DataFrame(
        {
            "path": [str(root / f"image_{index}.jpg") for index in range(6)],
            "y_true": [0, 0, 1, 1, 1, 1],
            "fake_score": source_scores,
        }
    ).to_csv(run_dir / "train" / "predictions.csv", index=False)


def _write_metadata(root: Path) -> Path:
    metadata = root / "metadata.csv"
    pd.DataFrame(
        {
            "path": [str(root / f"image_{index}.jpg") for index in range(6)],
            "label": [0, 0, 1, 1, 1, 1],
            "source_label": [0, 0, 1, 1, 2, 2],
        }
    ).to_csv(metadata, index=False)
    return metadata


def test_load_source_metadata_normalizes_paths(tmp_path: Path) -> None:
    metadata = _write_metadata(tmp_path)

    frame = load_source_metadata(metadata)

    assert {"path_key", "source_label", "label"}.issubset(frame.columns)
    assert frame["source_label"].tolist() == [0, 0, 1, 1, 2, 2]


def test_source_holdout_selection_uses_generator_folds_and_cap(tmp_path: Path) -> None:
    metadata = _write_metadata(tmp_path)
    _write_run(
        tmp_path,
        "source_pretty_overfires",
        seed=7,
        threshold=0.5,
        target_accuracy=0.60,
        target_predicted_positive_rate=0.90,
        source_scores=[0.1, 0.2, 0.8, 0.8, 0.8, 0.8],
    )
    _write_run(
        tmp_path,
        "capped_candidate",
        seed=7,
        threshold=0.75,
        target_accuracy=0.80,
        target_predicted_positive_rate=0.50,
        source_scores=[0.1, 0.2, 0.4, 0.8, 0.4, 0.8],
    )

    folds = collect_source_holdout_metrics(
        tmp_path,
        "ms_cocoai_to_ishu_neural_fusion",
        metadata,
        seeds=[7],
        candidates=["source_pretty_overfires", "capped_candidate"],
        variant="ishu_test",
    )
    unconstrained = select_candidates(folds, score_mode="mean", cap=None)
    capped = select_candidates(folds, score_mode="mean", cap=0.4)
    summary = summarize_selection(concat_selection_frames([unconstrained, capped]))
    source_summary = summarize_policy_source_holdouts(
        folds, concat_selection_frames([unconstrained, capped])
    )

    assert len(folds["heldout_source_label"].unique()) == 2
    assert unconstrained.iloc[0]["candidate"] == "source_pretty_overfires"
    assert capped.iloc[0]["candidate"] == "capped_candidate"
    capped_summary = summary[
        summary["selection_policy"] == "source_holdout_mean_utility_cap_0p4"
    ].iloc[0]
    assert capped_summary["target_accuracy_mean"] == 0.80
    assert capped_summary["target_predicted_positive_rate_mean"] == 0.50
    capped_source_summary = source_summary[
        source_summary["selection_policy"] == "source_holdout_mean_utility_cap_0p4"
    ]
    assert capped_source_summary["heldout_source_name"].tolist() == ["sd21", "sdxl"]
    assert capped_source_summary["source_holdout_recall_mean"].tolist() == [0.5, 0.5]
    assert capped_source_summary["source_holdout_fake_miss_rate_mean"].tolist() == [0.5, 0.5]


def test_source_decision_metrics_from_frame_scores_subsets() -> None:
    frame = pd.DataFrame(
        {
            "y_true": [0, 0, 1, 1],
            "fake_score": [0.2, 0.9, 0.4, 0.8],
        }
    )

    metrics = source_decision_metrics_from_frame(frame, threshold=0.5)

    assert metrics["recall"] == 0.5
    assert metrics["specificity"] == 0.5
    assert metrics["predicted_positive_rate"] == 0.5
