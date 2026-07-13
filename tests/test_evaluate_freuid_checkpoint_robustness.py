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


def test_robustness_evaluator_writes_paired_variant_metrics(tmp_path: Path) -> None:
    rows = []
    for index in range(8):
        path = tmp_path / f"sample_{index}.jpeg"
        Image.fromarray(np.full((40, 64, 3), 30 * index, dtype=np.uint8)).save(path)
        rows.append(
            {
                "id": path.stem,
                "label": index % 2,
                "type": "A/DL" if index < 4 else "B/ID",
                "local_path": str(path),
            }
        )
    data_csv = tmp_path / "data.csv"
    pd.DataFrame(rows).to_csv(data_csv, index=False)

    model = build_freuid_model("tiny_cnn", num_types=2, pretrained=False, forensic_residual=True)
    checkpoint = tmp_path / "model.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "model": "tiny_cnn",
            "image_size": 32,
            "type_to_idx": {"A/DL": 0, "B/ID": 1},
            "forensic_residual": True,
        },
        checkpoint,
    )
    output_dir = tmp_path / "out"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "evaluate_freuid_checkpoint_robustness.py"),
            "--checkpoint",
            str(checkpoint),
            "--data-csv",
            str(data_csv),
            "--output-dir",
            str(output_dir),
            "--variants",
            "clean",
            "jpeg50",
            "--batch-size",
            "4",
            "--num-workers",
            "0",
            "--device",
            "cpu",
        ],
        cwd=ROOT,
        check=True,
    )

    summary = pd.read_csv(output_dir / "summary.csv")
    assert set(summary["variant"]) == {"clean", "jpeg50"}
    assert (output_dir / "predictions_clean.csv").exists()
    assert (output_dir / "predictions_jpeg50.csv").exists()
