from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_method_family_comparison_writes_multi_objective_winners(tmp_path: Path) -> None:
    core_results = tmp_path / "publication_core_results.csv"
    csv_out = tmp_path / "method_family_comparison.csv"
    report_out = tmp_path / "method_family_comparison.md"
    rows = [_core_row(finding_id, method) for finding_id, method in _methods().items()]
    core = pd.DataFrame(rows)
    core.loc[core["finding_id"] == "ishu_same_physics_guided", ["accuracy", "auc"]] = [0.84, 0.91]
    core.loc[core["finding_id"] == "ishu_same_combined_v4_raw", "auc"] = 0.92
    core.loc[core["finding_id"] == "ishu_to_ms_clip_standalone", ["accuracy", "auc"]] = [0.70, 0.90]
    core.loc[core["finding_id"] == "ishu_to_ms_scp_fusion_clip", "brier"] = 0.19
    core.loc[core["finding_id"] == "ishu_to_ms_triage5_clip_standalone", ["coverage", "decided_accuracy"]] = [
        0.47,
        0.92,
    ]
    core.loc[
        core["finding_id"] == "ishu_to_ms_triage5_scp_fusion_all_foundation",
        ["coverage", "decided_accuracy"],
    ] = [0.31, 0.87]
    core.loc[core["finding_id"] == "ms_to_ishu_tuned_fusion_native_tiling_best", "accuracy"] = 0.78
    core.loc[core["finding_id"] == "ms_to_ishu_branch_dropout_auc", "auc"] = 0.86
    core.to_csv(core_results, index=False)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_method_family_comparison.py"),
            "--core-results",
            str(core_results),
            "--csv-out",
            str(csv_out),
            "--report-out",
            str(report_out),
        ],
        cwd=ROOT,
        check=True,
    )

    comparison = pd.read_csv(csv_out)
    report = report_out.read_text(encoding="utf-8")
    forward_auc = comparison[
        (comparison["scenario_id"] == "forward_transfer_ishu_to_ms")
        & (comparison["criterion"] == "best_auc")
    ].iloc[0]
    reverse_accuracy = comparison[
        (comparison["scenario_id"] == "reverse_transfer_ms_to_ishu")
        & (comparison["criterion"] == "best_accuracy")
    ].iloc[0]

    assert set(comparison["scenario_id"]) == {
        "same_domain_ishu",
        "forward_transfer_ishu_to_ms",
        "forward_triage_ishu_to_ms",
        "reverse_transfer_ms_to_ishu",
    }
    assert forward_auc["finding_id"] == "ishu_to_ms_clip_standalone"
    assert forward_auc["winner_family"] == "frozen foundation"
    assert reverse_accuracy["finding_id"] == "ms_to_ishu_tuned_fusion_native_tiling_best"
    assert reverse_accuracy["runner_up_margin"] > 0
    assert "Do not collapse these rows into a single overall winner" in report


def _core_row(finding_id: str, method: str) -> dict[str, object]:
    return {
        "finding_id": finding_id,
        "setting": "setting",
        "method": method,
        "accuracy": 0.60,
        "auc": 0.70,
        "brier": 0.30,
        "ece": 0.25,
        "predicted_fake_rate": 0.40,
        "coverage": pd.NA,
        "decided_accuracy": pd.NA,
        "source": "source",
        "interpretation": "interpretation",
    }


def _methods() -> dict[str, str]:
    return {
        "ishu_same_combined_v3": "combined_v3",
        "ishu_same_resnet18": "ResNet-18",
        "ishu_same_physics_guided": "Physics-guided ResNet-18 + combined_v3",
        "ishu_same_combined_v4_raw": "combined_v4",
        "ishu_same_combined_v4_selectk60": "combined_v4 select-k60",
        "ishu_to_ms_combined_v3": "combined_v3",
        "ishu_to_ms_resnet18": "ResNet-18",
        "ishu_to_ms_physics_guided": "Physics-guided ResNet-18 + combined_v3",
        "ishu_to_ms_convnext_tiny_frozen": "Frozen ConvNeXt-Tiny",
        "ishu_to_ms_combined_v4_raw": "combined_v4",
        "ishu_to_ms_combined_v4_selectk60": "combined_v4 select-k60",
        "ishu_to_ms_scp_fusion_v0": "SCP-Fusion v0",
        "ishu_to_ms_scp_fusion_dinov2": "SCP-Fusion + DINOv2",
        "ishu_to_ms_scp_fusion_clip": "SCP-Fusion + CLIP",
        "ishu_to_ms_scp_fusion_all_foundation": "SCP-Fusion + CLIP + DINOv2",
        "ishu_to_ms_clip_standalone": "Frozen CLIP ViT-B/32",
        "ishu_to_ms_triage5_scp_fusion_all_foundation": "SCP-Fusion + CLIP + DINOv2",
        "ishu_to_ms_triage5_clip_standalone": "Frozen CLIP ViT-B/32",
        "ms_to_ishu_physics_guided": "Physics-guided ResNet-18 + combined_v3",
        "ms_to_ishu_clip_vit_b_32": "Frozen CLIP ViT-B/32",
        "ms_to_ishu_score_fusion_all6_temp_balanced": "Reverse all-branch fusion + balanced temperature",
        "ms_to_ishu_branch_dropout_auc": "Reverse all-branch dropout fusion",
        "ms_to_ishu_source_cap_accuracy": "Reverse source-threshold capped fusion",
        "ms_to_ishu_source_utility_cap_0p48": "Reverse capped source-utility model selection",
        "ms_to_ishu_source_holdout_mean_utility_cap_0p48": "Reverse capped source-heldout utility selection",
        "ms_to_ishu_source_holdout_tuned_fusion": "Reverse source-heldout tuned fusion",
        "ms_to_ishu_tuned_fusion_constraint_sweep_best": "Reverse tuned-fusion constraint sweep best",
        "ms_to_ishu_tuned_fusion_native_tiling_best": "Reverse tuned fusion + native tiled combined_v3",
    }
