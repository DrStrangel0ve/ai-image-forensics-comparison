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
