from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from forensic_compare.photometric import FEATURE_NAMES, estimate_pseudo_normals, extract_features
from forensic_compare.conventional import (
    extract_feature_set,
    feature_names,
)


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


def test_conventional_feature_sets_are_finite(tmp_path: Path) -> None:
    image = np.random.default_rng(1).integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
    path = tmp_path / "image.png"
    Image.fromarray(image).save(path)

    for feature_set in ["noise", "noise_v2", "noise_v3", "combined", "combined_v2", "combined_v3"]:
        features = extract_feature_set(path, image_size=32, feature_set=feature_set)
        assert features.shape == (len(feature_names(feature_set)),)
        assert np.isfinite(features).all()

    assert len(feature_names("noise_v2")) > len(feature_names("noise"))
    assert len(feature_names("noise_v3")) > len(feature_names("noise_v2"))
    assert len(feature_names("combined_v2")) > len(feature_names("combined"))
    assert len(feature_names("combined_v3")) > len(feature_names("combined_v2"))
