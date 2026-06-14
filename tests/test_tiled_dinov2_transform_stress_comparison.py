from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_tiled_dinov2_transform_stress_comparison_builder_writes_deltas(tmp_path: Path) -> None:
    out_path = tmp_path / "tiled_dino_transform_stress.md"
    csv_out = tmp_path / "tiled_dino_transform_stress.csv"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_tiled_dinov2_transform_stress_comparison.py"),
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

    assert "Tiled DINO Transform Stress Comparison" in text
    assert "Largest accuracy lift" in text
    assert "tile_max" in text
    assert {"blur1", "jpeg30", "resize_half", "screenshot"} == set(comparison["variant"])
    assert (comparison["accuracy_delta"] > 0).all()
    assert (comparison["auc_delta"] > 0).all()
    assert "standalone_best_auc" in comparison.columns
    assert "source_report" in comparison.columns
