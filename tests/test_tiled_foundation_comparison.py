from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_tiled_foundation_comparison_builder_writes_encoder_winners(tmp_path: Path) -> None:
    out_path = tmp_path / "tiled_foundation_comparison.md"
    csv_out = tmp_path / "tiled_foundation_comparison.csv"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_tiled_foundation_comparison.py"),
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

    assert "Tiled Foundation Reverse-Transfer Comparison" in text
    assert "Best default-threshold accuracy: `ConvNeXt-Tiny`" in text
    assert "Best ranking AUC: `CLIP ViT-B/32`" in text
    assert "tile_top2_mean" in text
    assert {"CLIP ViT-B/32", "DINOv2-small", "ConvNeXt-Tiny"} == set(
        comparison["encoder"]
    )
    assert len(comparison) == 12
    assert "accuracy_mean_delta_vs_global" in comparison.columns
    best_accuracy = comparison.sort_values("accuracy_mean", ascending=False).iloc[0]
    best_auc = comparison.sort_values("roc_auc_mean", ascending=False).iloc[0]
    assert best_accuracy["encoder"] == "ConvNeXt-Tiny"
    assert best_accuracy["score_mode"] == "tile_mean"
    assert best_auc["encoder"] == "CLIP ViT-B/32"
    assert best_auc["score_mode"] == "tile_top2_mean"
