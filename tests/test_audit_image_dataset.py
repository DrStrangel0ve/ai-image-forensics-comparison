from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from scripts.audit_image_dataset import audit_dataset


def _write_image(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.random.default_rng(seed).integers(0, 255, size=(16, 18, 3), dtype=np.uint8)
    Image.fromarray(image).save(path)


def _write_split_pattern(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, : width // 2] = 255
    Image.fromarray(image).save(path)


def test_audit_dataset_reports_exact_cross_split_duplicates(tmp_path: Path) -> None:
    _write_image(tmp_path / "train" / "real" / "a.png", 1)
    _write_image(tmp_path / "train" / "fake" / "b.png", 2)
    _write_image(tmp_path / "valid" / "real" / "c.png", 3)
    duplicate_source = tmp_path / "train" / "fake" / "b.png"
    duplicate_target = tmp_path / "valid" / "fake" / "b-copy.png"
    duplicate_target.parent.mkdir(parents=True, exist_ok=True)
    duplicate_target.write_bytes(duplicate_source.read_bytes())

    summary = audit_dataset(tmp_path)

    assert summary["n_images"] == 4
    assert summary["n_duplicate_groups"] == 1
    assert summary["n_cross_split_duplicate_groups"] == 1
    assert summary["n_cross_class_duplicate_groups"] == 0
    assert summary["class_counts"] == [
        {"split": "test", "class_name": "fake", "label": 1, "n_images": 1},
        {"split": "test", "class_name": "real", "label": 0, "n_images": 1},
        {"split": "train", "class_name": "fake", "label": 1, "n_images": 1},
        {"split": "train", "class_name": "real", "label": 0, "n_images": 1},
    ]


def test_audit_dataset_reports_near_duplicate_resized_images(tmp_path: Path) -> None:
    source = tmp_path / "train" / "real" / "a.png"
    resized = tmp_path / "valid" / "real" / "a-resized.png"
    _write_split_pattern(source, (32, 32))
    resized.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        image.resize((24, 24), Image.Resampling.LANCZOS).save(resized)
    _write_image(tmp_path / "train" / "fake" / "b.png", 1)
    _write_image(tmp_path / "valid" / "fake" / "c.png", 2)

    summary = audit_dataset(tmp_path, near_duplicate_distance=2)

    assert summary["n_duplicate_groups"] == 0
    assert summary["n_near_duplicate_pairs"] >= 1
    assert summary["n_cross_split_near_duplicate_pairs"] >= 1
    assert any(row["cross_split"] for row in summary["near_duplicates"])
