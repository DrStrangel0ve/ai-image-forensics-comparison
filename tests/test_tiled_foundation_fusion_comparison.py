from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_tiled_foundation_fusion_comparison_builder_writes_frontiers(tmp_path: Path) -> None:
    out_path = tmp_path / "tiled_foundation_fusion.md"
    csv_out = tmp_path / "tiled_foundation_fusion.csv"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_tiled_foundation_fusion_comparison.py"),
            "--out-path",
            str(out_path),
            "--csv-out",
            str(csv_out),
        ],
        cwd=ROOT,
        check=True,
    )

    text = out_path.read_text(encoding="utf-8")
    comparison = pd.read_csv(csv_out)

    assert "MS COCOAI to Ishu Tiled Foundation Fusion Comparison" in text
    assert "Best accuracy-preserving tiled foundation replacement: `DINOv2-small`" in text
    assert "Best AUC-only tiled foundation replacement: `ConvNeXt-Tiny`" in text
    assert {"combined_v3", "clip_vit_b_32", "dinov2_vits14", "convnext_tiny"} == set(
        comparison["branch"]
    )
    assert len(comparison) == 16
    assert "target_roc_auc_mean_delta_vs_previous_tiled_v3" in comparison.columns

    best_accuracy = comparison.sort_values(
        ["target_accuracy_mean", "target_roc_auc_mean", "target_predicted_positive_rate_mean"],
        ascending=[False, False, True],
    ).iloc[0]
    best_auc = comparison.sort_values(
        ["target_roc_auc_mean", "target_accuracy_mean", "target_predicted_positive_rate_mean"],
        ascending=[False, False, True],
    ).iloc[0]

    assert best_accuracy["branch"] == "dinov2_vits14"
    assert best_accuracy["score_mode"] == "tile_top2_mean"
    assert best_auc["branch"] == "convnext_tiny"
    assert best_auc["score_mode"] == "tile_mean"
