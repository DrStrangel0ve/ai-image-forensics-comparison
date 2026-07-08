from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_fuse_freuid_scores_writes_grid_and_best_predictions(tmp_path: Path) -> None:
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    out_dir = tmp_path / "fusion"
    pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e", "f"],
            "y_true": [0, 0, 0, 1, 1, 1],
            "fraud_score": [0.05, 0.20, 0.30, 0.55, 0.65, 0.90],
        }
    ).to_csv(first, index=False)
    pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e", "f"],
            "y_true": [0, 0, 0, 1, 1, 1],
            "fraud_score": [0.10, 0.25, 0.35, 0.40, 0.85, 0.95],
        }
    ).to_csv(second, index=False)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "fuse_freuid_scores.py"),
            "--predictions",
            str(first),
            str(second),
            "--names",
            "first",
            "second",
            "--output-dir",
            str(out_dir),
            "--weight-step",
            "0.5",
            "--normalization",
            "raw",
            "rank",
        ],
        cwd=ROOT,
        check=True,
    )

    summary = json.loads((out_dir / "fusion_summary.json").read_text(encoding="utf-8"))
    grid = pd.read_csv(out_dir / "fusion_grid.csv")
    fused = pd.read_csv(out_dir / "fused_predictions.csv")
    assert summary["source_names"] == ["first", "second"]
    assert summary["n_rows"] == 6
    assert len(grid) == 6
    assert list(fused.columns) == ["id", "y_true", "fraud_score", "label"]
    assert set(fused["label"]).issubset({0, 1})
