from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from forensic_compare.datasets import class_kind, collect_labeled_images, discover_layout
from scripts.export_image_split import _records_for_split


def _image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8)).save(path)


def test_discover_train_test_layout(tmp_path: Path) -> None:
    _image(tmp_path / "train" / "REAL" / "a.png")
    _image(tmp_path / "train" / "FAKE" / "b.png")
    _image(tmp_path / "test" / "REAL" / "c.png")
    _image(tmp_path / "test" / "FAKE" / "d.png")

    layout = discover_layout(tmp_path)

    assert layout.train == tmp_path / "train"
    assert layout.test == tmp_path / "test"
    assert layout.single is None


def test_collect_labeled_images_uses_fake_as_one(tmp_path: Path) -> None:
    _image(tmp_path / "REAL" / "a.png")
    _image(tmp_path / "FAKE" / "b.png")

    records = collect_labeled_images(tmp_path)
    labels = sorted(record[1] for record in records)

    assert labels == [0, 1]
    assert class_kind("AI_generated") == "fake"
    assert class_kind("human-generated") == "real"
    assert class_kind("machine-generated") == "fake"
    assert class_kind("real") == "real"


def test_export_records_for_split_is_stratified(tmp_path: Path) -> None:
    for index in range(5):
        _image(tmp_path / "REAL" / f"real-{index}.png")
        _image(tmp_path / "FAKE" / f"fake-{index}.png")

    records, _layout = _records_for_split(tmp_path, split="test", val_fraction=0.4, seed=7)
    labels = sorted(record[1] for record in records)

    assert len(records) == 4
    assert labels == [0, 0, 1, 1]
