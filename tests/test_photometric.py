from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from forensic_compare.photometric import FEATURE_NAMES, estimate_pseudo_normals, extract_features
from forensic_compare.conventional import (
    extract_feature_set,
    feature_names,
)
from forensic_compare.transforms import ROBUSTNESS_VARIANTS, apply_robustness_variant
from scripts.make_robustness_variants import _apply_variant
from scripts.run_feature_baseline import _classifier, _extract_matrix, _selected_feature_names


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

    for feature_set in [
        "noise",
        "noise_v2",
        "noise_v3",
        "noise_v4",
        "combined",
        "combined_v2",
        "combined_v3",
        "combined_v4",
    ]:
        features = extract_feature_set(path, image_size=32, feature_set=feature_set)
        assert features.shape == (len(feature_names(feature_set)),)
        assert np.isfinite(features).all()

    assert len(feature_names("noise_v2")) > len(feature_names("noise"))
    assert len(feature_names("noise_v3")) > len(feature_names("noise_v2"))
    assert len(feature_names("noise_v4")) > len(feature_names("noise_v3"))
    assert len(feature_names("combined_v2")) > len(feature_names("combined"))
    assert len(feature_names("combined_v3")) > len(feature_names("combined_v2"))
    assert len(feature_names("combined_v4")) > len(feature_names("combined_v3"))


def test_robustness_variants_preserve_image_size() -> None:
    image = Image.fromarray(
        np.random.default_rng(2).integers(0, 255, size=(33, 41, 3), dtype=np.uint8)
    )

    for variant in ROBUSTNESS_VARIANTS:
        transformed = _apply_variant(image, variant)
        assert transformed.size == image.size
        assert transformed.mode == "RGB"


def test_shared_robustness_variants_match_script_wrapper() -> None:
    image = Image.fromarray(
        np.random.default_rng(3).integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
    )

    for variant in ROBUSTNESS_VARIANTS:
        shared = apply_robustness_variant(image, variant)
        wrapped = _apply_variant(image, variant)
        assert shared.size == wrapped.size == image.size
        assert shared.mode == wrapped.mode == "RGB"


def test_noise_robustness_variant_is_deterministic() -> None:
    image = Image.fromarray(
        np.random.default_rng(6).integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
    )

    first = np.asarray(apply_robustness_variant(image, "noise3"))
    second = np.asarray(apply_robustness_variant(image, "noise3"))

    assert np.array_equal(first, second)
    assert not np.array_equal(first, np.asarray(image.convert("RGB")))


def test_social_square_variant_preserves_nonsquare_canvas() -> None:
    image = Image.fromarray(
        np.random.default_rng(7).integers(0, 255, size=(24, 40, 3), dtype=np.uint8)
    )

    transformed = apply_robustness_variant(image, "social_square")

    assert transformed.size == image.size
    assert transformed.mode == "RGB"
    assert not np.array_equal(np.asarray(transformed), np.asarray(image.convert("RGB")))


def test_feature_extraction_can_augment_training_rows(tmp_path: Path) -> None:
    image = np.random.default_rng(4).integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
    path = tmp_path / "image.png"
    Image.fromarray(image).save(path)

    features, labels, paths, skipped = _extract_matrix(
        [(path, 1, "ai_generated")],
        image_size=32,
        feature_set="noise",
        desc="test/features",
        skip_errors=False,
        augment_variants=["blur1", "resize_half"],
    )

    assert features.shape == (3, len(feature_names("noise")))
    assert labels.tolist() == [1, 1, 1]
    assert paths == [path, path, path]
    assert skipped == []
    assert np.isfinite(features).all()


def test_feature_classifier_can_select_top_k_features() -> None:
    rng = np.random.default_rng(5)
    x_train = rng.normal(size=(18, 6))
    y_train = np.array([0, 1] * 9)
    names = [f"feature_{index}" for index in range(x_train.shape[1])]

    classifier = _classifier(
        "logistic_regression",
        seed=5,
        select_k=3,
        n_features=x_train.shape[1],
    )
    classifier.fit(x_train, y_train)

    selected = _selected_feature_names(classifier, names)
    assert len(selected) == 3
    assert set(selected).issubset(names)
