from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _summary(path: Path, raw_auc: float, select_ece: float) -> None:
    rows = [
        {
            "run": "combined_v3_logreg",
            "feature_set": "combined_v3",
            "classifier": "logistic_regression",
            "select_k": 0,
            "n_runs": 3,
            "accuracy_mean": 0.70,
            "accuracy_ci_low": 0.68,
            "accuracy_ci_high": 0.72,
            "roc_auc_mean": 0.80,
            "roc_auc_ci_low": 0.78,
            "roc_auc_ci_high": 0.82,
            "brier_score_mean": 0.20,
            "brier_score_ci_low": 0.19,
            "brier_score_ci_high": 0.21,
            "expected_calibration_error_mean": 0.12,
            "expected_calibration_error_ci_low": 0.10,
            "expected_calibration_error_ci_high": 0.14,
        },
        {
            "run": "combined_v4_logreg",
            "feature_set": "combined_v4",
            "classifier": "logistic_regression",
            "select_k": 0,
            "n_runs": 3,
            "accuracy_mean": 0.73,
            "accuracy_ci_low": 0.70,
            "accuracy_ci_high": 0.75,
            "roc_auc_mean": raw_auc,
            "roc_auc_ci_low": raw_auc - 0.02,
            "roc_auc_ci_high": raw_auc + 0.02,
            "brier_score_mean": 0.19,
            "brier_score_ci_low": 0.18,
            "brier_score_ci_high": 0.20,
            "expected_calibration_error_mean": 0.11,
            "expected_calibration_error_ci_low": 0.09,
            "expected_calibration_error_ci_high": 0.13,
        },
        {
            "run": "combined_v4_logreg_selectk60",
            "feature_set": "combined_v4",
            "classifier": "logistic_regression",
            "select_k": 60,
            "n_runs": 3,
            "accuracy_mean": 0.72,
            "accuracy_ci_low": 0.70,
            "accuracy_ci_high": 0.74,
            "roc_auc_mean": 0.82,
            "roc_auc_ci_low": 0.80,
            "roc_auc_ci_high": 0.84,
            "brier_score_mean": 0.18,
            "brier_score_ci_low": 0.17,
            "brier_score_ci_high": 0.19,
            "expected_calibration_error_mean": select_ece,
            "expected_calibration_error_ci_low": select_ece - 0.01,
            "expected_calibration_error_ci_high": select_ece + 0.01,
        },
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def test_combined_v4_transfer_readiness_builder_outputs_report_tables_and_commands(
    tmp_path: Path,
) -> None:
    medium = tmp_path / "medium.csv"
    small = tmp_path / "small.csv"
    core = tmp_path / "publication_core_results.csv"
    report = tmp_path / "report.md"
    table = tmp_path / "readiness.csv"
    commands = tmp_path / "commands.csv"

    _summary(medium, raw_auc=0.84, select_ece=0.08)
    _summary(small, raw_auc=0.79, select_ece=0.13)
    pd.DataFrame(
        {
            "finding_id": ["ishu_to_ms_combined_v3"],
            "setting": ["transfer"],
            "method": ["combined_v3"],
            "accuracy": [0.55],
            "auc": [0.58],
            "brier": [0.34],
            "ece": [0.29],
        }
    ).to_csv(core, index=False)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_combined_v4_transfer_readiness.py"),
            "--medium-summary",
            str(medium),
            "--small-summary",
            str(small),
            "--core-results",
            str(core),
            "--out-path",
            str(report),
            "--table-out",
            str(table),
            "--commands-out",
            str(commands),
            "--seeds",
            "7",
            "17",
        ],
        cwd=ROOT,
        check=True,
    )

    text = report.read_text(encoding="utf-8")
    readiness = pd.read_csv(table)
    command_manifest = pd.read_csv(commands)

    assert "combined_v4 Transfer Readiness" in text
    assert "No `combined_v4` transfer row" in text
    assert "combined_v4_logreg_selectk60" in set(readiness["candidate"])
    assert len(command_manifest) == 12
    assert set(command_manifest["phase"]) == {"train", "transfer_eval"}
    assert "evaluate_feature_model.py" in command_manifest.iloc[-1]["command"]
