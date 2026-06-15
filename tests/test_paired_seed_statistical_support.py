from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_paired_seed_statistical_support_builds_guarded_report(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    ishu_physics = repo_root / "ishu_physics.csv"
    combined_v4 = repo_root / "combined_v4.csv"
    score_fusion = repo_root / "score_fusion.csv"
    reverse_transfer = repo_root / "reverse.csv"
    report_out = repo_root / "reports" / "paired_seed_statistical_support_2026_06_15.md"
    csv_out = repo_root / "reports" / "assets" / "paired_seed_statistical_support.csv"

    pd.DataFrame(_ishu_physics_rows()).to_csv(ishu_physics, index=False)
    pd.DataFrame(_combined_v4_rows()).to_csv(combined_v4, index=False)
    pd.DataFrame(_score_fusion_rows()).to_csv(score_fusion, index=False)
    pd.DataFrame(_reverse_rows()).to_csv(reverse_transfer, index=False)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_paired_seed_statistical_support.py"),
            "--repo-root",
            str(repo_root),
            "--ishu-physics-runs",
            str(ishu_physics.relative_to(repo_root)),
            "--combined-v4",
            str(combined_v4.relative_to(repo_root)),
            "--score-fusion",
            str(score_fusion.relative_to(repo_root)),
            "--reverse-transfer",
            str(reverse_transfer.relative_to(repo_root)),
            "--report-out",
            str(report_out.relative_to(repo_root)),
            "--csv-out",
            str(csv_out.relative_to(repo_root)),
            "--run-date",
            "2026-06-15",
        ],
        cwd=ROOT,
        check=True,
    )

    report = report_out.read_text(encoding="utf-8")
    support = pd.read_csv(csv_out)
    physics_acc = support[
        (support["comparison_id"] == "ishu_physics_guided_vs_resnet18")
        & (support["metric"] == "accuracy")
    ].iloc[0]
    calibrated_brier = support[
        (support["comparison_id"] == "ishu_to_ms_source_calibrated_all_foundation_vs_clip")
        & (support["metric"] == "Brier")
    ].iloc[0]
    diagnostic_rows = support[support["direction"] == "diagnostic"]

    assert "Status: **PASS**" in report
    assert "not as definitive hypothesis tests" in report
    assert len(support) == 39
    assert physics_acc["support_label"] == "consistent_gain_ci_excludes_zero"
    assert int(physics_acc["candidate_wins"]) == 3
    assert calibrated_brier["support_label"] == "consistent_gain_ci_excludes_zero"
    assert set(diagnostic_rows["support_label"]) == {"diagnostic_shift"}


def _ishu_physics_rows() -> list[dict[str, object]]:
    rows = []
    for seed, resnet_acc, physics_acc, combined_acc in [
        ("seed1", 0.70, 0.75, 0.74),
        ("seed2", 0.72, 0.78, 0.79),
        ("seed3", 0.71, 0.77, 0.73),
    ]:
        rows.extend(
            [
                {
                    "run": seed,
                    "method": "neural_resnet18",
                    "accuracy": resnet_acc,
                    "roc_auc": resnet_acc + 0.10,
                },
                {
                    "run": seed,
                    "method": "physics_guided_resnet18_combined_v3",
                    "accuracy": physics_acc,
                    "roc_auc": physics_acc + 0.10,
                },
                {
                    "run": seed,
                    "method": "feature_combined_v3",
                    "accuracy": combined_acc,
                    "roc_auc": combined_acc + 0.10,
                },
            ]
        )
    return rows


def _combined_v4_rows() -> list[dict[str, object]]:
    rows = []
    for phase in ["ishu_holdout", "ishu_to_ms_cocoai"]:
        for seed in [1, 2, 3]:
            rows.extend(
                [
                    _combined_row(phase, seed, "combined_v3_logreg", 0.60, 0.70, 0.30, 0.20, 0.45),
                    _combined_row(phase, seed, "combined_v4_logreg", 0.59, 0.71, 0.31, 0.22, 0.46),
                    _combined_row(
                        phase,
                        seed,
                        "combined_v4_logreg_selectk60",
                        0.62,
                        0.73,
                        0.28,
                        0.18,
                        0.44,
                    ),
                ]
            )
    return rows


def _combined_row(
    phase: str,
    seed: int,
    run: str,
    accuracy: float,
    auc: float,
    brier: float,
    ece: float,
    fake_rate: float,
) -> dict[str, object]:
    jitter = seed * 0.001
    return {
        "phase": phase,
        "seed": seed,
        "run": run,
        "accuracy": accuracy + jitter,
        "roc_auc": auc + jitter,
        "brier_score": brier - jitter,
        "expected_calibration_error": ece - jitter,
        "fake_call_rate": fake_rate + jitter,
    }


def _score_fusion_rows() -> list[dict[str, object]]:
    rows = []
    methods = {
        "clip_standalone": (0.64, 0.86, 0.32, 0.33, 0.16),
        "scp_fusion_all_foundation": (0.62, 0.80, 0.31, 0.32, 0.13),
        "all_foundation_source_calibrated": (0.63, 0.79, 0.29, 0.30, 0.15),
        "clip_source_calibrated": (0.63, 0.78, 0.30, 0.31, 0.15),
    }
    for seed in ["seed1", "seed2", "seed3"]:
        for method, values in methods.items():
            accuracy, auc, brier, ece, positive_rate = values
            rows.append(
                {
                    "group": seed,
                    "method": method,
                    "accuracy": accuracy,
                    "roc_auc": auc,
                    "brier_score": brier,
                    "expected_calibration_error": ece,
                    "predicted_positive_rate": positive_rate,
                }
            )
    return rows


def _reverse_rows() -> list[dict[str, object]]:
    rows = []
    methods = {
        "resnet18": (0.61, 0.70, 0.30, 0.27, 0.58),
        "physics_guided_resnet18_combined_v3": (0.67, 0.74, 0.24, 0.19, 0.53),
        "clip_vit_b_32": (0.62, 0.82, 0.33, 0.35, 0.86),
        "score_fusion_all6_temp_balanced": (0.66, 0.83, 0.30, 0.32, 0.81),
    }
    for seed in [1, 2, 3]:
        for method, values in methods.items():
            accuracy, auc, brier, ece, fake_rate = values
            rows.append(
                {
                    "split": "ms_cocoai_to_ishu_test",
                    "seed": seed,
                    "method": method,
                    "accuracy": accuracy,
                    "auc": auc,
                    "brier": brier,
                    "ece": ece,
                    "predicted_fake_rate": fake_rate,
                }
            )
    return rows
