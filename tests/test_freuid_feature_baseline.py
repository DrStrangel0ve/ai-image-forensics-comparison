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
    test_csv = tmp_path / "sample_submission.csv"
    output_dir = tmp_path / "run"
    cache_dir = tmp_path / "cache"
    test_predictions = tmp_path / "test_predictions.csv"
    pd.DataFrame(train_rows).to_csv(train_csv, index=False)
    pd.DataFrame(val_rows).to_csv(val_csv, index=False)
    for idx in range(2):
        image_id = f"test_{idx}"
        _write_image(image_root / "public_test" / "public_test" / f"{image_id}.jpeg", idx + 200)
    pd.DataFrame({"id": ["test_0", "test_1"], "label": [0, 0]}).to_csv(test_csv, index=False)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_freuid_feature_baseline.py"),
            "--train-csv",
            str(train_csv),
            "--val-csv",
            str(val_csv),
            "--test-csv",
            str(test_csv),
            "--image-root",
            str(image_root),
            "--output-dir",
            str(output_dir),
            "--test-predictions-out",
            str(test_predictions),
            "--feature-set",
            "photometric",
            "--image-size",
            "32",
            "--limit-balance-columns",
            "type",
            "label",
            "--feature-cache-dir",
            str(cache_dir),
        ],
        cwd=ROOT,
        check=True,
    )

    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    predictions = pd.read_csv(output_dir / "val_predictions.csv")
    public_predictions = pd.read_csv(test_predictions)
    assert metrics["method"] == "freuid_feature_baseline"
    assert metrics["n_train"] == 8
    assert metrics["n_val"] == 4
    assert metrics["n_test"] == 2
    assert metrics["feature_cache"] == {
        "train_hits": 0,
        "train_misses": 8,
        "val_hits": 0,
        "val_misses": 4,
        "test_hits": 0,
        "test_misses": 2,
    }
    assert len(list(cache_dir.rglob("*.npy"))) == 14
    assert "apcer_at_1pct_bpcer" in metrics
    assert list(predictions.columns) == ["id", "image_path", "local_path", "type", "y_true", "fraud_score", "label"]
    assert list(public_predictions.columns) == ["id", "image_path", "local_path", "type", "fraud_score"]
    assert list(public_predictions["id"]) == ["test_0", "test_1"]
