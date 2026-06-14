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
        "feature_set": "reconstruction_v2",
        "method": "feature_reconstruction_v2_logistic_regression",
        "n_samples": 20,
        "n_test": 20,
        "n_train": 80,
        "precision": 0.78,
        "recall": 0.70,
        "threshold": 0.5,
    }
    metrics.update(overrides)
    path.write_text(json.dumps(metrics), encoding="utf-8")


def test_reconstruction_v2_probe_summary_outputs_same_and_transfer_report(
    tmp_path: Path,
) -> None:
    roots = {
        "v2": tmp_path / "v2",
        "lite": tmp_path / "lite",
        "combined": tmp_path / "combined",
        "v2_transfer": tmp_path / "v2_transfer",
        "lite_transfer": tmp_path / "lite_transfer",
        "combined_transfer": tmp_path / "combined_transfer",
    }
    for seed in [7, 17]:
        _write_metrics(roots["v2"] / f"seed{seed}" / "metrics.json", roc_auc=0.80)
        _write_metrics(
            roots["lite"] / f"seed{seed}" / "metrics.json",
            feature_set="reconstruction_lite",
            roc_auc=0.76,
        )
        _write_metrics(
            roots["combined"] / f"seed{seed}" / "metrics.json",
            feature_set="combined_v3",
            roc_auc=0.82,
        )
        _write_metrics(
            roots["v2_transfer"] / f"seed{seed}" / "metrics.json",
            method="cross_feature_reconstruction_v2",
            n_target=100,
            roc_auc=0.65,
        )
        _write_metrics(
            roots["lite_transfer"] / f"seed{seed}" / "metrics.json",
            feature_set="reconstruction_lite",
            method="cross_feature_reconstruction_lite",
            n_target=100,
            roc_auc=0.68,
        )
        _write_metrics(
            roots["combined_transfer"] / f"seed{seed}" / "metrics.json",
            feature_set="combined_v3",
            method="cross_feature_combined_v3",
            n_target=100,
            roc_auc=0.61,
        )

    seed_out = tmp_path / "seed.csv"
    mean_out = tmp_path / "mean.csv"
    delta_out = tmp_path / "delta.csv"
    report_out = tmp_path / "report.md"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "summarize_reconstruction_v2_probe.py"),
            "--v2-root",
            str(roots["v2"]),
            "--lite-root",
            str(roots["lite"]),
            "--combined-root",
            str(roots["combined"]),
            "--v2-transfer-root",
            str(roots["v2_transfer"]),
            "--lite-transfer-root",
            str(roots["lite_transfer"]),
            "--combined-transfer-root",
            str(roots["combined_transfer"]),
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

    assert set(seed_summary["setting"]) == {"ishu_same_bounded", "ishu_to_ms_cocoai_bounded"}
    assert set(mean_summary["candidate"]) == {
        "combined_v3_logreg",
        "reconstruction_lite_logreg",
        "reconstruction_v2_logreg",
    }
    assert "reconstruction_v2 Probe" in report
    transfer_delta = delta_summary[
        (delta_summary["setting"] == "ishu_to_ms_cocoai_bounded")
        & (delta_summary["baseline"] == "reconstruction_lite_logreg")
        & (delta_summary["metric"] == "roc_auc")
    ]["delta_mean"].item()
    assert round(float(transfer_delta), 2) == -0.03
