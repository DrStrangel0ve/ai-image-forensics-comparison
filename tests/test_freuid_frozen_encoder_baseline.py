from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def _image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full((20, 20, 3), value, dtype=np.uint8)
    Image.fromarray(array).save(path)


def _metadata(ids: list[str], labels: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": ids,
            "image_path": [f"train/{image_id}.jpeg" for image_id in ids],
            "label": labels,
            "is_digital": [True] * len(ids),
            "type": ["BENIN/DL", "BENIN/DL", "EGYPT/DL", "EGYPT/DL"][: len(ids)],
        }
    )


def test_run_freuid_frozen_encoder_baseline_writes_predictions(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    train_ids = ["train_real_0", "train_fake_0", "train_real_1", "train_fake_1"]
    val_ids = ["val_real_0", "val_fake_0", "val_real_1", "val_fake_1"]
    for image_id, label in zip(train_ids + val_ids, [0, 1, 0, 1, 0, 1, 0, 1]):
        _image(image_root / "train" / "train" / f"{image_id}.jpeg", 32 if label == 0 else 220)

    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    _metadata(train_ids, [0, 1, 0, 1]).to_csv(train_csv, index=False)
    _metadata(val_ids, [0, 1, 0, 1]).to_csv(val_csv, index=False)
    out_dir = tmp_path / "freuid_frozen"
    cache_dir = tmp_path / "embedding_cache"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_freuid_frozen_encoder_baseline.py"),
            "--train-csv",
            str(train_csv),
            "--val-csv",
            str(val_csv),
            "--image-root",
            str(image_root),
            "--output-dir",
            str(out_dir),
            "--encoder",
            "tiny_cnn",
            "--classifier",
            "logistic_regression",
            "--batch-size",
            "2",
            "--device",
            "cpu",
            "--embedding-cache-dir",
            str(cache_dir),
        ],
        cwd=ROOT,
        check=True,
    )

    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(out_dir / "val_predictions.csv")
    assert metrics["method"] == "freuid_frozen_encoder_tiny_cnn_logistic_regression"
    assert metrics["embedding_dim"] == 128
    assert metrics["n_train"] == 4
    assert metrics["n_val"] == 4
    assert metrics["embedding_cache"]["train_misses"] == 4
    assert set(["id", "image_path", "local_path", "type", "y_true", "fraud_score", "label"]).issubset(
        predictions.columns
    )
    assert (out_dir / "classifier.joblib").exists()
    assert (out_dir / "encoder.json").exists()
    assert (out_dir / "embeddings.npz").exists()
