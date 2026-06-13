from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
METHODS = [
    "combined_v3",
    "resnet18",
    "physics_guided",
    "convnext_tiny_frozen",
    "scp_fusion_v0",
]


def test_publication_asset_builder_writes_expected_figures(tmp_path: Path) -> None:
    calibration = tmp_path / "calibration.csv"
    source_calibration = tmp_path / "source_calibration.csv"
    triage_5 = tmp_path / "triage_5.csv"
    triage_10 = tmp_path / "triage_10.csv"
    tuned_triage = tmp_path / "tuned_triage.csv"
    dino_calibration = tmp_path / "dino_calibration.csv"
    dino_triage_5 = tmp_path / "dino_triage_5.csv"
    dino_triage_10 = tmp_path / "dino_triage_10.csv"
    clip_calibration = tmp_path / "clip_calibration.csv"
    clip_triage_5 = tmp_path / "clip_triage_5.csv"
    clip_triage_10 = tmp_path / "clip_triage_10.csv"
    reverse_fusion = tmp_path / "reverse_fusion.csv"
    reverse_thresholds = tmp_path / "reverse_thresholds.csv"
    reverse_all_metrics = tmp_path / "reverse_all_metrics.csv"
    reverse_all_thresholds = tmp_path / "reverse_all_thresholds.csv"
    reverse_source_threshold_fusion = tmp_path / "reverse_source_threshold_fusion.csv"
    reverse_threshold_tiebreak = tmp_path / "reverse_threshold_tiebreak.csv"
    reverse_threshold_cap = tmp_path / "reverse_threshold_cap.csv"
    out_dir = tmp_path / "assets"

    pd.DataFrame(
        {
            "method": METHODS,
            "mean_accuracy": [0.54, 0.58, 0.61, 0.62, 0.59],
            "mean_roc_auc": [0.58, 0.65, 0.66, 0.71, 0.73],
            "mean_brier_score": [0.34, 0.35, 0.33, 0.32, 0.31],
            "mean_expected_calibration_error": [0.29, 0.33, 0.30, 0.31, 0.30],
        }
    ).to_csv(calibration, index=False)

    pd.DataFrame(
        {
            "method": METHODS,
            "calibrator": ["temperature_balanced"] * len(METHODS),
            "mean_raw_brier_score": [0.24, 0.27, 0.26, 0.22, 0.20],
            "mean_calibrated_brier_score": [0.23, 0.22, 0.21, 0.20, 0.19],
            "mean_raw_ece": [0.18, 0.24, 0.23, 0.19, 0.15],
            "mean_calibrated_ece": [0.18, 0.15, 0.15, 0.14, 0.13],
        }
    ).to_csv(source_calibration, index=False)

    base_triage = pd.DataFrame(
        {
            "method": METHODS * 2,
            "score_mode": ["raw"] * len(METHODS) + ["temperature_balanced"] * len(METHODS),
            "mean_test_coverage": [0.10, 0.18, 0.19, 0.24, 0.21] * 2,
            "mean_test_triage_accuracy": [0.60, 0.69, 0.66, 0.75, 0.74] * 2,
        }
    )
    base_triage.to_csv(triage_5, index=False)
    triage_10_frame = base_triage.copy()
    triage_10_frame["mean_test_coverage"] = triage_10_frame["mean_test_coverage"] + 0.2
    triage_10_frame["mean_test_triage_accuracy"] = triage_10_frame["mean_test_triage_accuracy"] - 0.05
    triage_10_frame.to_csv(triage_10, index=False)

    pd.DataFrame(
        {
            "method": ["scp_fusion_v0", "branch_dropout", "source_calibrated"],
            "selected_score_modes": ["raw:15", "raw:15", "raw:15"],
            "mean_test_utility": [0.11, 0.09, 0.08],
            "mean_test_utility_ci_low": [0.04, 0.02, 0.01],
            "mean_test_utility_ci_high": [0.18, 0.16, 0.15],
            "mean_test_fake_detection": [0.40, 0.38, 0.37],
            "mean_test_real_clearance": [0.36, 0.34, 0.35],
            "mean_test_real_fpr": [0.12, 0.12, 0.12],
            "mean_test_fake_false_clearance": [0.15, 0.15, 0.15],
        }
    ).to_csv(tuned_triage, index=False)

    dino_methods = [
        "scp_fusion_v0",
        "source_calibrated",
        "scp_fusion_dinov2",
        "dinov2_source_calibrated",
    ]
    pd.DataFrame(
        {
            "method": dino_methods,
            "mean_accuracy": [0.59, 0.61, 0.60, 0.62],
            "mean_roc_auc": [0.73, 0.72, 0.75, 0.75],
            "mean_brier_score": [0.32, 0.31, 0.31, 0.30],
        }
    ).to_csv(dino_calibration, index=False)
    for path, coverage_base, accuracy_base in [
        (dino_triage_5, 0.20, 0.75),
        (dino_triage_10, 0.43, 0.71),
    ]:
        pd.DataFrame(
            {
                "method": dino_methods * 2,
                "score_mode": ["raw"] * len(dino_methods)
                + ["temperature_balanced"] * len(dino_methods),
                "mean_test_coverage": [coverage_base, coverage_base, coverage_base + 0.05, coverage_base + 0.04]
                * 2,
                "mean_test_triage_accuracy": [
                    accuracy_base,
                    accuracy_base - 0.02,
                    accuracy_base + 0.05,
                    accuracy_base + 0.06,
                ]
                * 2,
            }
        ).to_csv(path, index=False)

    clip_methods = [
        "scp_fusion_v0",
        "scp_fusion_dinov2",
        "scp_fusion_clip",
        "scp_fusion_all_foundation",
        "clip_standalone",
    ]
    pd.DataFrame(
        {
            "method": clip_methods,
            "mean_accuracy": [0.59, 0.60, 0.62, 0.62, 0.64],
            "mean_roc_auc": [0.73, 0.75, 0.79, 0.80, 0.86],
        }
    ).to_csv(clip_calibration, index=False)
    for path, coverage_base, accuracy_base in [
        (clip_triage_5, 0.21, 0.75),
        (clip_triage_10, 0.44, 0.72),
    ]:
        pd.DataFrame(
            {
                "method": clip_methods * 2,
                "score_mode": ["raw"] * len(clip_methods)
                + ["temperature_balanced"] * len(clip_methods),
                "mean_test_coverage": [
                    coverage_base,
                    coverage_base + 0.04,
                    coverage_base + 0.08,
                    coverage_base + 0.10,
                    coverage_base + 0.26,
                ]
                * 2,
                "mean_test_triage_accuracy": [
                    accuracy_base,
                    accuracy_base + 0.05,
                    accuracy_base + 0.10,
                    accuracy_base + 0.12,
                    accuracy_base + 0.18,
                ]
                * 2,
            }
        ).to_csv(path, index=False)

    reverse_configs = [
        "score_fusion_all6",
        "score_fusion_all6_temp_balanced",
        "score_fusion_all6_c01",
        "score_fusion_all6_c01_temp_balanced",
        "score_fusion_all6_c003",
        "score_fusion_all6_dropout_mean_r35x8",
        "score_fusion_all6_dropout_mean_r35x8_temp_balanced",
    ]
    pd.DataFrame(
        {
            "config": reverse_configs * 2,
            "split": ["ms_cocoai_validation"] * len(reverse_configs)
            + ["ms_cocoai_to_ishu_test"] * len(reverse_configs),
            "accuracy": [0.95] * len(reverse_configs)
            + [0.652, 0.658, 0.649, 0.658, 0.661, 0.652, 0.652],
            "auc": [0.99] * len(reverse_configs)
            + [0.827, 0.829, 0.837, 0.832, 0.831, 0.841, 0.840],
            "brier": [0.03] * len(reverse_configs)
            + [0.294, 0.307, 0.247, 0.279, 0.221, 0.306, 0.294],
        }
    ).to_csv(reverse_fusion, index=False)
    pd.DataFrame(
        {
            "config": reverse_configs,
            "default_accuracy": [0.652, 0.658, 0.649, 0.658, 0.661, 0.652, 0.652],
            "clean_threshold_accuracy": [0.664, 0.664, 0.675, 0.684, 0.667, 0.658, 0.643],
        }
    ).to_csv(reverse_thresholds, index=False)
    reverse_method_metrics = [
        ("physics_guided_resnet18_combined_v3", 0.687, 0.742, 0.244, 0.190, 0.529),
        ("convnext_tiny", 0.658, 0.808, 0.286, 0.295, 0.734),
        ("clip_vit_b_32", 0.623, 0.824, 0.332, 0.357, 0.863),
        ("score_fusion_all6_temp_balanced", 0.658, 0.829, 0.307, 0.330, 0.810),
    ]
    pd.DataFrame(
        [
            {
                "method": method,
                "split": "ms_cocoai_to_ishu_test",
                "accuracy": accuracy,
                "auc": auc,
                "brier": brier,
                "ece": ece,
                "predicted_fake_rate": predicted_fake_rate,
            }
            for method, accuracy, auc, brier, ece, predicted_fake_rate in reverse_method_metrics
        ]
    ).to_csv(reverse_all_metrics, index=False)
    pd.DataFrame(
        {
            "method": [
                "physics_guided",
                "convnext_tiny",
                "clip_vit_b_32",
                "score_fusion_all6_temp_balanced",
            ],
            "split": ["ms_cocoai_to_ishu_test"] * 4,
            "default_accuracy": [0.687, 0.658, 0.623, 0.658],
            "clean_threshold_accuracy": [0.681, 0.678, 0.646, 0.664],
            "auc": [0.742, 0.808, 0.824, 0.829],
        }
    ).to_csv(reverse_all_thresholds, index=False)
    pd.DataFrame(
        {
            "config": ["score_fusion_all6_c003_source_acc"],
            "variant": ["ishu_test"],
            "accuracy": [0.696],
            "auc": [0.829],
            "brier": [0.219],
            "ece": [0.206],
            "predicted_fake_rate": [0.708],
        }
    ).to_csv(reverse_source_threshold_fusion, index=False)
    pd.DataFrame(
        {
            "config": ["score_fusion_all6_c003_source_acc"],
            "threshold_tiebreak": ["higher"],
            "variant": ["ishu_test"],
            "accuracy": [0.719],
            "auc": [0.829],
            "brier": [0.219],
            "ece": [0.206],
            "predicted_fake_rate": [0.661],
        }
    ).to_csv(reverse_threshold_tiebreak, index=False)
    pd.DataFrame(
        {
            "config": ["cap_0p48"],
            "variant": ["ishu_test"],
            "accuracy": [0.722],
            "auc": [0.829],
            "brier": [0.219],
            "ece": [0.206],
            "predicted_fake_rate": [0.623],
        }
    ).to_csv(reverse_threshold_cap, index=False)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_publication_assets.py"),
            "--calibration-summary",
            str(calibration),
            "--source-heldout-calibration",
            str(source_calibration),
            "--triage-5pct",
            str(triage_5),
            "--triage-10pct",
            str(triage_10),
            "--score-fusion-tuned-triage",
            str(tuned_triage),
            "--score-fusion-dinov2-calibration",
            str(dino_calibration),
            "--score-fusion-dinov2-triage-5pct",
            str(dino_triage_5),
            "--score-fusion-dinov2-triage-10pct",
            str(dino_triage_10),
            "--score-fusion-clip-calibration",
            str(clip_calibration),
            "--score-fusion-clip-triage-5pct",
            str(clip_triage_5),
            "--score-fusion-clip-triage-10pct",
            str(clip_triage_10),
            "--reverse-fusion-regularization",
            str(reverse_fusion),
            "--reverse-fusion-thresholds",
            str(reverse_thresholds),
            "--reverse-all-method-metrics",
            str(reverse_all_metrics),
            "--reverse-all-method-thresholds",
            str(reverse_all_thresholds),
            "--reverse-source-threshold-fusion",
            str(reverse_source_threshold_fusion),
            "--reverse-threshold-tiebreak",
            str(reverse_threshold_tiebreak),
            "--reverse-threshold-cap",
            str(reverse_threshold_cap),
            "--out-dir",
            str(out_dir),
            "--dpi",
            "80",
        ],
        cwd=ROOT,
        check=True,
    )

    assert (out_dir / "publication_cross_domain_calibration.png").exists()
    assert (out_dir / "publication_source_heldout_calibration.png").exists()
    assert (out_dir / "publication_triage_operating_points.png").exists()
    assert (out_dir / "publication_score_fusion_tuned_triage.png").exists()
    assert (out_dir / "publication_score_fusion_dinov2_gain.png").exists()
    assert (out_dir / "publication_score_fusion_clip_frontier.png").exists()
    assert (out_dir / "publication_reverse_fusion_tradeoff.png").exists()
    assert (out_dir / "publication_reverse_operating_points.png").exists()
