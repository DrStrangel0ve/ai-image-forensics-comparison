from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_freuid_timm_probe import aggregate_view_embeddings, limit_frame  # noqa: E402


def test_aggregate_view_embeddings_exposes_local_statistics() -> None:
    embeddings = np.asarray(
        [
            [[1.0, 2.0], [3.0, 0.0]],
            [[2.0, 4.0], [2.0, 4.0]],
        ],
        dtype=np.float32,
    )

    assert aggregate_view_embeddings(embeddings, "mean").shape == (2, 2)
    assert aggregate_view_embeddings(embeddings, "mean_max").shape == (2, 4)
    combined = aggregate_view_embeddings(embeddings, "mean_max_std")
    assert combined.shape == (2, 6)
    assert np.allclose(combined[0], [2.0, 1.0, 3.0, 2.0, 1.0, 1.0])


def test_limit_frame_retains_each_type_label_group() -> None:
    frame = pd.DataFrame(
        [
            {"id": f"{doc_type}-{label}-{index}", "type": doc_type, "label": label}
            for doc_type in ("A/DL", "B/ID")
            for label in (0, 1)
            for index in range(4)
        ]
    )

    limited = limit_frame(frame, max_samples=8, seed=43)

    assert len(limited) == 8
    assert limited.groupby(["type", "label"]).size().to_dict() == {
        ("A/DL", 0): 2,
        ("A/DL", 1): 2,
        ("B/ID", 0): 2,
        ("B/ID", 1): 2,
    }


def test_tiny_five_crop_probe_runs_end_to_end(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    image_root.mkdir()
    rows = []
    for index in range(12):
        label = index % 2
        doc_type = "A/DL" if index < 8 else "B/ID"
        image_id = f"sample_{index}"
        pixels = np.full((64, 96, 3), 40 + 100 * label + index, dtype=np.uint8)
        Image.fromarray(pixels).save(image_root / f"{image_id}.jpeg")
        rows.append(
            {
                "id": image_id,
                "image_path": f"{image_id}.jpeg",
                "label": label,
                "type": doc_type,
            }
        )
    train_csv = tmp_path / "train.csv"
    val_csv = tmp_path / "val.csv"
    pd.DataFrame(rows[:8]).to_csv(train_csv, index=False)
    pd.DataFrame(rows[8:]).to_csv(val_csv, index=False)
    output_dir = tmp_path / "output"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "run_freuid_timm_probe.py"),
            "--train-csv",
            str(train_csv),
            "--val-csv",
            str(val_csv),
            "--image-root",
            str(image_root),
            "--output-dir",
            str(output_dir),
            "--model",
            "tiny_cnn",
            "--no-pretrained",
            "--image-size",
            "64",
            "--view-mode",
            "five_crop",
            "--batch-size",
            "2",
            "--view-batch-size",
            "4",
            "--num-workers",
            "0",
            "--device",
            "cpu",
            "--aggregation",
            "mean",
            "mean_max_std",
            "--primary-aggregation",
            "mean_max_std",
            "--max-iterations",
            "50",
        ],
        cwd=ROOT,
        check=True,
    )

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["competition_eligibility"] == "post_freeze_research_only"
    assert summary["n_train"] == 8
    assert summary["n_val"] == 4
    assert summary["n_views"] == 5
    assert (output_dir / "train_embeddings.npz").exists()
    assert (output_dir / "val_predictions_mean_max_std.csv").exists()
