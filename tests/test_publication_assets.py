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
