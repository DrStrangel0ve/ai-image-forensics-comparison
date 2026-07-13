from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from forensic_compare.freuid_model import build_freuid_model


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from infer_freuid_checkpoint_ensemble import rank_percentiles  # noqa: E402


def _checkpoint(path: Path, bias: float) -> None:
    model = build_freuid_model("tiny_cnn", num_types=2, pretrained=False)
    with torch.no_grad():
        model.fraud_head[1].weight.zero_()
        model.fraud_head[1].bias.fill_(bias)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model": "tiny_cnn",
            "image_size": 64,
            "type_to_idx": {"A/DL": 0, "B/ID": 1},
            "threshold": 0.5,
        },
        path,
    )


def test_rank_percentiles_uses_average_ranks_for_ties() -> None:
    actual = rank_percentiles(np.asarray([10.0, 20.0, 20.0, 40.0]))
    np.testing.assert_allclose(actual, [0.25, 0.625, 0.625, 1.0])


def test_checkpoint_ensemble_writes_scores_and_hash_manifest(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    for index, value in enumerate([20, 90, 180]):
        Image.fromarray(np.full((40, 64, 3), value, dtype=np.uint8)).save(image_dir / f"sample_{index}.png")
    first = tmp_path / "first.pt"
    second = tmp_path / "second.pt"
    _checkpoint(first, -1.0)
    _checkpoint(second, 1.0)

    output = tmp_path / "submission.csv"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "infer_freuid_checkpoint_ensemble.py"),
            "--input-dir",
            str(image_dir),
            "--checkpoint",
            str(first),
            "--checkpoint",
            str(second),
            "--weight",
            "0.85",
            "--weight",
            "0.15",
            "--normalization",
            "raw",
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
    assert frame.shape == (3, 2)
    expected = 0.85 / (1.0 + np.exp(1.0)) + 0.15 / (1.0 + np.exp(-1.0))
    np.testing.assert_allclose(frame["label"], expected, atol=1e-9)
    manifest = json.loads(output.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert manifest["runtime"] == "sequential_checkpoint_ensemble"
    assert [member["weight"] for member in manifest["members"]] == [0.85, 0.15]
    assert all(len(member["sha256"]) == 64 for member in manifest["members"])
