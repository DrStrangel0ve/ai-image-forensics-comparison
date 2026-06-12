from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from forensic_compare.foundation import build_frozen_encoder, supported_frozen_encoders


ROOT = Path(__file__).resolve().parents[1]


def _image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.full((16, 16, 3), value, dtype=np.uint8)
    Image.fromarray(array).save(path)


def test_frozen_encoder_baseline_writes_standard_outputs(tmp_path: Path) -> None:
    for split in ["train", "test"]:
        for index in range(3):
            _image(tmp_path / split / "REAL" / f"real-{index}.png", 32)
            _image(tmp_path / split / "FAKE" / f"fake-{index}.png", 220)

    out_dir = tmp_path / "frozen_encoder"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_frozen_encoder_baseline.py"),
            "--data-dir",
            str(tmp_path),
            "--output-dir",
            str(out_dir),
            "--encoder",
            "tiny_cnn",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--seed",
            "3",
        ],
        cwd=ROOT,
        check=True,
    )

    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["method"] == "frozen_encoder_tiny_cnn_logistic_regression"
    assert metrics["embedding_dim"] == 128
    assert metrics["n_train"] == 6
    assert metrics["n_test"] == 6
    assert (out_dir / "predictions.csv").exists()
    assert (out_dir / "classifier.joblib").exists()
    assert (out_dir / "encoder.json").exists()
    assert (out_dir / "embeddings.npz").exists()

    eval_dir = tmp_path / "frozen_encoder_eval"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_frozen_encoder_model.py"),
            "--model-dir",
            str(out_dir),
            "--target-dir",
            str(tmp_path),
            "--output-dir",
            str(eval_dir),
            "--target-split",
            "all",
            "--batch-size",
            "2",
            "--num-workers",
            "0",
            "--device",
            "cpu",
        ],
        cwd=ROOT,
        check=True,
    )

    eval_metrics = json.loads((eval_dir / "metrics.json").read_text(encoding="utf-8"))
    assert eval_metrics["method"] == "cross_frozen_encoder"
    assert eval_metrics["n_target"] == 12
    assert (eval_dir / "predictions.csv").exists()


def test_foundation_encoder_registry_exposes_clip_and_dino_aliases() -> None:
    encoders = set(supported_frozen_encoders())
    assert "clip_vit_b_32" in encoders
    assert "dinov2_vits14" in encoders
    assert "dinov2_vitb14" in encoders
    with pytest.raises(ValueError, match="requires --pretrained"):
        build_frozen_encoder("clip_vit_b_32", pretrained=False)
