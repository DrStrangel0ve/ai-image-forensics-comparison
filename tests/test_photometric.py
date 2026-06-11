from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from forensic_compare.photometric import FEATURE_NAMES, estimate_pseudo_normals, extract_features


def test_extract_features_are_finite(tmp_path: Path) -> None:
    gradient = np.linspace(0, 255, 32, dtype=np.uint8)
    image = np.tile(gradient[None, :, None], (32, 1, 3))
    path = tmp_path / "gradient.png"
    Image.fromarray(image).save(path)

    features = extract_features(path, image_size=32)

    assert features.shape == (len(FEATURE_NAMES),)
    assert np.isfinite(features).all()


def test_pseudo_normals_are_unit_length() -> None:
    gray = np.random.default_rng(0).random((16, 16), dtype=np.float32)
    normals = estimate_pseudo_normals(gray)
    lengths = np.linalg.norm(normals, axis=-1)

    assert np.allclose(lengths, 1.0, atol=1e-5)
