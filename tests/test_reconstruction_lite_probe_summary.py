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
        "accuracy": 0.70,
        "roc_auc": 0.75,
        "brier_score": 0.20,
        "classifier": "logistic_regression",
        "confusion_matrix": [[8, 2], [3, 7]],
        "expected_calibration_error": 0.10,
        "f1": 0.74,
        "feature_set": "reconstruction_lite",
        "method": "feature_reconstruction_lite_logistic_regression",
        "n_test": 20,
        "n_train": 80,
        "precision": 0.78,
        "recall": 0.70,
        "select_k": 0,
        "threshold": 0.5,
    }
    metrics.update(overrides)
    path.write_text(json.dumps(metrics), encoding="utf-8")


def test_reconstruction_lite_probe_summary_outputs_seed_mean_delta_and_report(
    tmp_path: Path,
) -> None:
    reconstruction = tmp_path / "reconstruction"
    combined = tmp_path / "combined"
    seed_out = tmp_path / "seed.csv"
    mean_out = tmp_path / "mean.csv"
    delta_out = tmp_path / "delta.csv"
    report_out = tmp_path / "report.md"

    _write_metrics(reconstruction / "seed7" / "metrics.json", accuracy=0.70, roc_auc=0.75)
    _write_metrics(reconstruction / "seed17" / "metrics.json", accuracy=0.72, roc_auc=0.77)
    _write_metrics(
        combined / "seed7" / "metrics.json",
        accuracy=0.80,
        roc_auc=0.86,
        feature_set="combined_v3",
        method="feature_combined_v3_logistic_regression",
    )
    _write_metrics(
        combined / "seed17" / "metrics.json",
        accuracy=0.82,
        roc_auc=0.88,
        feature_set="combined_v3",
        method="feature_combined_v3_logistic_regression",
    )

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "summarize_reconstruction_lite_probe.py"),
            "--reconstruction-root",
            str(reconstruction),
            "--combined-root",
            str(combined),
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

    assert len(seed_summary) == 4
    assert set(mean_summary["candidate"]) == {"combined_v3_logreg", "reconstruction_lite_logreg"}
    assert "reconstruction_lite Bounded Probe" in report
    assert "not a full benchmark claim" in report
    auc_delta = delta_summary[delta_summary["metric"] == "roc_auc"]["delta_mean"].item()
    assert round(float(auc_delta), 2) == -0.11
