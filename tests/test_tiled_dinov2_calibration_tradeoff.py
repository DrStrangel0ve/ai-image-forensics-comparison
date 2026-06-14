from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_tiled_dinov2_calibration_tradeoff_builder_writes_mode_split(tmp_path: Path) -> None:
    out_path = tmp_path / "calibration_tradeoff.md"
    csv_out = tmp_path / "calibration_tradeoff.csv"
    choices_out = tmp_path / "calibration_tradeoff_choices.csv"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_tiled_dinov2_calibration_tradeoff.py"),
            "--out-path",
            str(out_path),
            "--csv-out",
            str(csv_out),
            "--choice-csv-out",
            str(choices_out),
        ],
        cwd=ROOT,
        check=True,
    )

    text = out_path.read_text(encoding="utf-8")
    tradeoff = pd.read_csv(csv_out)
    choices = pd.read_csv(choices_out)

    assert "Tiled DINO Calibration Tradeoff" in text
    assert "Decision/ranking mode" in text
    assert "Calibration-safe mode" in text
    assert {"blur1", "jpeg30", "resize_half", "screenshot"} == set(tradeoff["variant"])
    assert set(choices["best_accuracy_mode"]) == {"tile_max"}
    assert (choices["best_accuracy_delta"] > 0).all()
    assert (choices["best_accuracy_auc_delta"] > 0).all()
    assert "target_brier_score_mean_delta_vs_global" in tradeoff.columns
    tile_mean = tradeoff[tradeoff["score_mode"] == "tile_mean"]
    assert (tile_mean["target_brier_score_mean_delta_vs_global"] < 0).all()
    assert (tile_mean["target_roc_auc_mean_delta_vs_global"] > 0).all()

