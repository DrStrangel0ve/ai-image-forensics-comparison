from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from forensic_compare.freuid_model import build_freuid_model


ROOT = Path(__file__).resolve().parents[1]


def test_finetuned_freuid_inference_writes_score_submission(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for index in range(3):
        pixels = np.full((40, 64, 3), 50 + 20 * index, dtype=np.uint8)
        Image.fromarray(pixels).save(image_dir / f"sample_{index}.jpeg")

    model = build_freuid_model("tiny_cnn", num_types=2, pretrained=False)
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model": "tiny_cnn",
            "image_size": 64,
            "type_to_idx": {"A/DL": 0, "B/ID": 1},
            "threshold": 0.4,
        },
        checkpoint,
    )
    output = tmp_path / "submission.csv"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "infer_freuid_finetune.py"),
            "--input-dir",
            str(image_dir),
            "--checkpoint",
            str(checkpoint),
            "--output-csv",
            str(output),
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

    frame = pd.read_csv(output)
    assert list(frame.columns) == ["id", "label"]
    assert list(frame["id"]) == ["sample_0", "sample_1", "sample_2"]
    assert frame["label"].between(0.0, 1.0).all()
    assert output.with_suffix(".manifest.json").exists()
