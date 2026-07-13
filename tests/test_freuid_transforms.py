from __future__ import annotations

import numpy as np
from PIL import Image

from forensic_compare.freuid_transforms import DocumentViewTransform, build_document_transforms


def test_document_view_transform_preserves_global_and_grid_views() -> None:
    image = Image.fromarray(np.full((80, 160, 3), 127, dtype=np.uint8))
    transform = DocumentViewTransform(size=48, grid_rows=2, grid_cols=3)

    tensor = transform(image)

    assert tensor.shape == (7, 3, 48, 48)


def test_single_document_view_remains_chw_for_checkpoint_compatibility() -> None:
    image = Image.fromarray(np.full((80, 160, 3), 127, dtype=np.uint8))
    train, evaluate = build_document_transforms(image_size=32, capture_strength=0.0)

    assert train(image).shape == (3, 32, 32)
    assert evaluate(image).shape == (3, 32, 32)


def test_five_crop_document_view_has_fixed_corner_and_center_views() -> None:
    image = Image.fromarray(np.full((80, 160, 3), 127, dtype=np.uint8))
    transform = DocumentViewTransform(size=48, view_mode="five_crop", five_crop_zoom=1.15)

    tensor = transform(image)

    assert tensor.shape == (5, 3, 48, 48)
