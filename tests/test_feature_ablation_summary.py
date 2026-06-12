from __future__ import annotations

from pathlib import Path

import pandas as pd

from forensic_compare.utils import write_json
from scripts.summarize_feature_ablation import summarize


def _write_metrics(path: Path, **overrides) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "method": "feature_combined_v4_logistic_regression_selectk2",
        "feature_set": "combined_v4",
        "classifier": "logistic_regression",
        "effective_select_k": 2,
        "selection_score_func": "f_classif",
        "n_train": 8,
        "n_samples": 4,
        "accuracy": 0.75,
        "precision": 0.8,
        "recall": 0.7,
        "f1": 0.74,
        "roc_auc": 0.9,
        "brier_score": 0.2,
        "expected_calibration_error": 0.1,
        "selected_feature_names": ["gray_mean", "fft_ring_00_10_ratio"],
    }
    metrics.update(overrides)
    write_json(metrics, path)


def test_summarize_feature_ablation_outputs_tables(tmp_path: Path) -> None:
    run_root = tmp_path / "runs"
    out_dir = tmp_path / "summary"
    _write_metrics(run_root / "seed7" / "combined_v4_selectk2" / "metrics.json")
    _write_metrics(
        run_root / "seed17" / "combined_v4_selectk2" / "metrics.json",
        accuracy=0.85,
        roc_auc=0.8,
    )

    summarize(run_root, out_dir, "combined_v3")

    summary = pd.read_csv(out_dir / "feature_ablation_summary.csv")
    selected = pd.read_csv(out_dir / "selected_feature_frequency.csv")

    assert summary.loc[0, "n_runs"] == 2
    assert summary.loc[0, "accuracy_mean"] == 0.8
    assert "accuracy_ci_low" in summary.columns
    assert "accuracy_ci_high" in summary.columns
    assert "roc_auc_ci_low" in summary.columns
    assert "roc_auc_ci_high" in summary.columns
    extra = selected[selected["feature"] == "fft_ring_00_10_ratio"].iloc[0]
    assert bool(extra["is_extra_feature"])
    assert extra["count"] == 2
    assert (out_dir / "report.md").exists()
