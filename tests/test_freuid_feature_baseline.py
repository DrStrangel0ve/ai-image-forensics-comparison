from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def _write_image(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    array = rng.integers(0, 255, size=(48, 48, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array, mode="RGB").save(path)


def test_run_freuid_feature_baseline_on_tiny_csv(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    train_rows = []
    val_rows = []
    for idx in range(8):
        image_id = f"train_{idx}"
        label = idx % 2
        image_path = f"train/{image_id}.jpeg"
        _write_image(image_root / "train" / "train" / f"{image_id}.jpeg", idx)
        train_rows.append({"id": image_id, "image_path": image_path, "label": label, "type": "A/DL"})
    for idx in range(4):
        image_id = f"val_{idx}"
        label = idx % 2
        image_path = f"train/{image_id}.jpeg"
        _write_image(image_root / "train" / "train" / f"{image_id}.jpeg", idx + 100)
        val_rows.append({"id": image_id, "image_path": image_path, "label": label, "type": "A/DL"})

    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    output_dir = tmp_path / "run"
    pd.DataFrame(train_rows).to_csv(train_csv, index=False)
    pd.DataFrame(val_rows).to_csv(val_csv, index=False)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_freuid_feature_baseline.py"),
            "--train-csv",
            str(train_csv),
            "--val-csv",
            str(val_csv),
            "--image-root",
            str(image_root),
            "--output-dir",
            str(output_dir),
            "--feature-set",
            "photometric",
            "--image-size",
            "32",
        ],
        cwd=ROOT,
        check=True,
    )

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(output_dir / "val_predictions.csv")
    assert metrics["method"] == "freuid_feature_baseline"
    assert metrics["n_train"] == 8
    assert metrics["n_val"] == 4
    assert "apcer_at_1pct_bpcer" in metrics
    assert list(predictions.columns) == ["id", "image_path", "local_path", "type", "y_true", "fraud_score", "label"]
