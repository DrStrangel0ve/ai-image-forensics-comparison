from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _write_metrics(path: Path, **overrides: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "accuracy": 0.60,
        "roc_auc": 0.70,
        "brier_score": 0.20,
        "classifier": "logistic_regression",
        "confusion_matrix": [[8, 2], [3, 7]],
        "expected_calibration_error": 0.10,
        "f1": 0.74,
        "feature_set": "combined_v3",
        "method": "feature_combined_v3_logistic_regression",
        "n_test": 20,
        "n_train": 80,
        "precision": 0.78,
        "recall": 0.70,
        "select_k": 0,
        "threshold": 0.5,
    }
    metrics.update(overrides)
    path.write_text(json.dumps(metrics), encoding="utf-8")


def test_combined_v4_transfer_summary_outputs_seed_mean_delta_and_report(tmp_path: Path) -> None:
    source = tmp_path / "source"
    transfer = tmp_path / "transfer"
    seed_out = tmp_path / "seed.csv"
    mean_out = tmp_path / "mean.csv"
    delta_out = tmp_path / "delta.csv"
    report_out = tmp_path / "report.md"

    for root in [source, transfer]:
        _write_metrics(root / "seed7" / "combined_v3_logreg" / "metrics.json")
        _write_metrics(
            root / "seed7" / "combined_v4_logreg" / "metrics.json",
            accuracy=0.63,
            roc_auc=0.74,
            brier_score=0.18,
            expected_calibration_error=0.08,
            feature_set="combined_v4",
            method="feature_combined_v4_logistic_regression",
        )
        _write_metrics(
            root / "seed7" / "combined_v4_logreg_selectk60" / "metrics.json",
            accuracy=0.65,
            roc_auc=0.76,
            brier_score=0.17,
            expected_calibration_error=0.07,
            feature_set="combined_v4",
            method="feature_combined_v4_logistic_regression_selectk60",
            effective_select_k=60,
            select_k=60,
        )

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "summarize_combined_v4_transfer.py"),
            "--source-root",
            str(source),
            "--transfer-root",
            str(transfer),
            "--seed-out",
            str(seed_out),
            "--mean-out",
            str(mean_out),
            "--delta-out",
            str(delta_out),
            "--report-out",
            str(report_out),
        ],
        cwd=ROOT,
        check=True,
    )

    seed_summary = pd.read_csv(seed_out)
    mean_summary = pd.read_csv(mean_out)
    delta_summary = pd.read_csv(delta_out)
    report = report_out.read_text(encoding="utf-8")

    assert len(seed_summary) == 6
    assert set(seed_summary["phase"]) == {"ishu_holdout", "ishu_to_ms_cocoai"}
    assert mean_summary["n_seeds"].eq(1).all()
    assert "combined_v4_logreg_selectk60" in set(delta_summary["candidate"])
    assert round(float(delta_summary["accuracy_delta_mean"].max()), 2) == 0.05
    assert "preliminary seed slice" in report
