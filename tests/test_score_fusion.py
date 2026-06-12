from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _predictions(path: Path, scores: list[float], labels: list[int]) -> None:
    frame = pd.DataFrame(
        {
            "path": [str(path.parent / f"image-{index}.jpg") for index in range(len(scores))],
            "y_true": labels,
            "fake_score": scores,
        }
    )
    frame.to_csv(path, index=False)


def test_score_fusion_writes_train_and_variant_outputs(tmp_path: Path) -> None:
    labels = [0, 0, 1, 1, 0, 1]
    _predictions(tmp_path / "train_a.csv", [0.1, 0.3, 0.6, 0.8, 0.2, 0.7], labels)
    _predictions(tmp_path / "train_b.csv", [0.2, 0.1, 0.7, 0.6, 0.3, 0.8], labels)
    _predictions(tmp_path / "target_a.csv", [0.2, 0.4, 0.7, 0.9, 0.1, 0.6], labels)
    _predictions(tmp_path / "target_b.csv", [0.1, 0.2, 0.8, 0.7, 0.2, 0.9], labels)
    out_dir = tmp_path / "fusion"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "fuse_prediction_scores.py"),
            "--out-dir",
            str(out_dir),
            "--train",
            f"a={tmp_path / 'train_a.csv'}",
            "--train",
            f"b={tmp_path / 'train_b.csv'}",
            "--variant",
            f"target:a={tmp_path / 'target_a.csv'}",
            "--variant",
            f"target:b={tmp_path / 'target_b.csv'}",
            "--seed",
            "3",
        ],
        cwd=ROOT,
        check=True,
    )

    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["base_methods"] == ["a", "b"]
    assert [row["variant"] for row in metrics["metrics"]] == ["train", "target"]
    assert (out_dir / "train" / "predictions.csv").exists()
    assert (out_dir / "target" / "predictions.csv").exists()
    assert (out_dir / "score_fusion_model.joblib").exists()
