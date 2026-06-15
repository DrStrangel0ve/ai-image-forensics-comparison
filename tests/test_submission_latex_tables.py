from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_submission_latex_table_builder_writes_escaped_fragments(tmp_path: Path) -> None:
    out_dir = tmp_path / "latex"
    report_out = tmp_path / "submission_latex_tables.md"
    source_dir = tmp_path / "assets"
    source_dir.mkdir()
    manifest_path = source_dir / "submission_result_table_manifest.csv"
    method_family_path = source_dir / "method_family_comparison.csv"
    calibration_modes_path = source_dir / "calibration_operating_modes.csv"
    paired_seed_path = source_dir / "paired_seed_statistical_support.csv"
    table_ids = [
        "same_domain_anchor",
        "transfer_frontier",
        "reverse_operating_points",
        "robustness_stress",
        "source_holdout_stress",
        "reconstruction_ablation",
    ]
    source_paths = []
    for table_id in table_ids:
        path = source_dir / f"{table_id}.csv"
        if table_id == "source_holdout_stress":
            pd.DataFrame(
                {
                    "selection_policy": ["source_holdout_mean_utility_cap_0p48"],
                    "heldout_source": ["sd3"],
                    "utility": [1.42345],
                    "recall": [0.79678],
                    "fake_miss_rate": [0.20322],
                    "predicted_fake_rate": [0.13254],
                    "paper_use": ["paper use"],
                }
            ).to_csv(path, index=False)
        elif table_id == "reconstruction_ablation":
            pd.DataFrame(
                {
                    "setting": ["ishu_same_bounded", "ishu_same_bounded"],
                    "setting_label": ["Ishu same-domain", "Ishu same-domain"],
                    "candidate": ["reconstruction_lite_logreg", "reconstruction_v2_logreg"],
                    "method": ["reconstruction_lite", "reconstruction_v2"],
                    "accuracy": [0.69, 0.71],
                    "auc": [0.73, 0.76],
                    "delta_auc_vs_reconstruction_lite": [0.0, 0.03],
                    "brier": [0.21, 0.20],
                    "ece": [0.10, 0.12],
                    "paper_use": ["paper use", "paper use"],
                }
            ).to_csv(path, index=False)
        else:
            pd.DataFrame(
                {
                    "finding_id": ["ishu_same_combined_v3", "ms_to_ishu_tuned_fusion_native_tiling_best"],
                    "method": ["combined_v3", "SCP-Fusion + native_v3"],
                    "accuracy": [0.81234, 0.75678],
                    "auc": [0.82345, 0.85678],
                    "brier": [0.2, 0.3],
                    "ece": [0.1, 0.2],
                    "fake_call_rate": [0.4, 0.5],
                    "coverage": [pd.NA, 0.25],
                    "decided_accuracy": [pd.NA, 0.9],
                    "delta_accuracy_vs_clean": [0.0, 0.05],
                    "delta_auc_vs_clean": [0.0, -0.01],
                }
            ).to_csv(path, index=False)
        source_paths.append(path.as_posix())
    pd.DataFrame(
        {
            "table_id": table_ids,
            "title": ["title"] * len(table_ids),
            "path": source_paths,
            "venue_use": ["WIFS,DFF"] * len(table_ids),
            "purpose": ["purpose"] * len(table_ids),
            "n_rows": [2] * len(table_ids),
        }
    ).to_csv(manifest_path, index=False)
    pd.DataFrame(
        {
            "scenario_id": ["forward_transfer_ishu_to_ms"],
            "scenario": ["Ishu -> MS COCOAI Transfer"],
            "criterion": ["best_auc"],
            "criterion_label": ["Best AUC"],
            "winner_family": ["frozen foundation"],
            "finding_id": ["ishu_to_ms_clip_standalone"],
            "method": ["Frozen CLIP ViT-B/32"],
        }
    ).to_csv(method_family_path, index=False)
    pd.DataFrame(
        [
            {
                "objective": "ranking_auc",
                "selected_method": "Frozen CLIP ViT-B/32",
                "selected_mode": "default_score",
                "metric": "auc",
                "metric_value": 0.8641,
            },
            {
                "objective": "probability_brier",
                "selected_method": "SCP-Fusion + CLIP",
                "selected_mode": "default_score",
                "metric": "brier",
                "metric_value": 0.3112,
            },
            {
                "objective": "reliability_ece",
                "selected_method": "combined_v4 select-k60",
                "selected_mode": "default_score",
                "metric": "ece",
                "metric_value": 0.2663,
            },
            {
                "objective": "tiled_dino_accuracy",
                "selected_method": "tiled DINOv2 reverse fusion",
                "selected_mode": "tile_max",
                "metric": "best_accuracy_delta",
                "metric_value": 0.0139,
            },
        ]
    ).to_csv(calibration_modes_path, index=False)
    pd.DataFrame(
        [
            _paired_row("ishu_physics_guided_vs_resnet18", "accuracy", 0.0205, 0.0088, 0.0263, 3),
            _paired_row("ishu_physics_guided_vs_resnet18", "AUC", 0.0250, 0.0160, 0.0391, 3),
            _paired_row("combined_v4_ishu_to_ms_selectk60_vs_v3", "AUC", 0.0119, -0.0233, 0.0365, 2),
            _paired_row("combined_v4_ishu_to_ms_selectk60_vs_v3", "ECE", -0.0249, -0.0499, -0.0062, 3),
            _paired_row("ishu_to_ms_scp_all_foundation_vs_clip", "AUC", -0.0646, -0.0749, -0.0475, 0, "candidate_trails"),
            _paired_row(
                "ishu_to_ms_source_calibrated_all_foundation_vs_clip",
                "Brier",
                -0.0161,
                -0.0219,
                -0.0096,
                3,
            ),
            _paired_row(
                "ishu_to_ms_source_calibrated_all_foundation_vs_clip",
                "ECE",
                -0.0237,
                -0.0364,
                -0.0164,
                3,
            ),
            _paired_row("ms_to_ishu_physics_guided_vs_resnet18", "accuracy", 0.0468, 0.0175, 0.1053, 3),
            _paired_row("ms_to_ishu_physics_guided_vs_resnet18", "ECE", -0.0770, -0.1040, -0.0406, 3),
            _paired_row("ms_to_ishu_temp_balanced_fusion_vs_clip", "accuracy", 0.0351, 0.0088, 0.0614, 3),
            _paired_row("ms_to_ishu_temp_balanced_fusion_vs_clip", "ECE", -0.0262, -0.0427, -0.0007, 3),
        ]
    ).to_csv(paired_seed_path, index=False)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_submission_latex_tables.py"),
            "--table-manifest",
            str(manifest_path),
            "--method-family-comparison",
            str(method_family_path),
            "--calibration-operating-modes",
            str(calibration_modes_path),
            "--paired-seed-support",
            str(paired_seed_path),
            "--out-dir",
            str(out_dir),
            "--report-out",
            str(report_out),
        ],
        cwd=ROOT,
        check=True,
    )

    latex_manifest = pd.read_csv(out_dir / "submission_latex_table_manifest.csv")
    same_tex = (out_dir / "same_domain_anchor.tex").read_text(encoding="utf-8")
    robustness_tex = (out_dir / "robustness_stress.tex").read_text(encoding="utf-8")
    source_tex = (out_dir / "source_holdout_stress.tex").read_text(encoding="utf-8")
    reconstruction_tex = (out_dir / "reconstruction_ablation.tex").read_text(encoding="utf-8")
    family_tex = (out_dir / "method_family_comparison.tex").read_text(encoding="utf-8")
    modes_tex = (out_dir / "calibration_operating_modes.tex").read_text(encoding="utf-8")
    paired_tex = (out_dir / "paired_seed_support.tex").read_text(encoding="utf-8")
    report = report_out.read_text(encoding="utf-8")

    assert set(latex_manifest["table_id"]) == set(table_ids) | {
        "method_family_comparison",
        "calibration_operating_modes",
        "paired_seed_support",
    }
    assert "\\begin{table}[t]" in same_tex
    assert "\\toprule" in same_tex
    assert "combined\\_v3" in same_tex
    assert "+0.050" in robustness_tex
    assert "-0.010" in robustness_tex
    assert "sd3" in source_tex
    assert "tab:source-holdout-stress" in source_tex
    assert "tab:reconstruction-ablation" in reconstruction_tex
    assert "reconstruction\\_v2" in reconstruction_tex
    assert "+0.030" in reconstruction_tex
    assert "tab:method-family-comparison" in family_tex
    assert "CLIP" in family_tex
    assert "tab:calibration-operating-modes" in modes_tex
    assert "Transfer ranking" in modes_tex
    assert "Frozen CLIP" in modes_tex
    assert "SCP-Fusion + CLIP" in modes_tex
    assert "combined\\_v4 k60" in modes_tex
    assert "dAcc." in modes_tex
    assert "tab:paired-seed-support" in paired_tex
    assert "Ishu: physics-guided vs ResNet" in paired_tex
    assert "+0.0205" in paired_tex
    assert "3/3" in paired_tex
    assert "candidate trails" in paired_tex
    assert "Submission LaTeX Tables" in report


def _paired_row(
    comparison_id: str,
    metric: str,
    delta: float,
    ci_low: float,
    ci_high: float,
    wins: int,
    label: str = "consistent_gain_ci_excludes_zero",
) -> dict[str, object]:
    return {
        "comparison_id": comparison_id,
        "comparison": comparison_id,
        "metric": metric,
        "direction": "higher",
        "candidate": "candidate",
        "baseline": "baseline",
        "n_paired_seeds": 3,
        "seeds": "seed1;seed2;seed3",
        "candidate_mean": 0.6 + delta,
        "baseline_mean": 0.6,
        "raw_delta_mean": delta,
        "raw_delta_ci_low": ci_low,
        "raw_delta_ci_high": ci_high,
        "favorable_delta_mean": delta,
        "favorable_delta_ci_low": ci_low,
        "favorable_delta_ci_high": ci_high,
        "candidate_wins": wins,
        "candidate_losses": 3 - wins,
        "ties": 0,
        "support_label": label,
        "note": "fixture",
    }
