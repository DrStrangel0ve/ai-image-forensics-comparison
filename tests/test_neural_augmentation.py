from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from torchvision import datasets, transforms

from scripts.train_neural_net import DeterministicAugmentedDataset


def _write_image(path: Path, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.random.default_rng(seed).integers(0, 255, size=(24, 24, 3), dtype=np.uint8)
    Image.fromarray(image).save(path)


def test_neural_train_dataset_can_apply_deterministic_variants(tmp_path: Path) -> None:
    _write_image(tmp_path / "real" / "a.png", 1)
    _write_image(tmp_path / "ai_generated" / "b.png", 2)
    base = datasets.ImageFolder(tmp_path, transform=transforms.ToTensor())

    augmented = DeterministicAugmentedDataset(base, variants=["blur1", "resize_half"])

    assert len(augmented) == 3 * len(base)
    image, label = augmented[len(base)]
    assert image.shape == (3, 24, 24)
    assert label in set(base.class_to_idx.values())
