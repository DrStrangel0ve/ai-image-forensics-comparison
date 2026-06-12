import numpy as np
import pytest
import torch

from scripts import evaluate_physics_guided_net as physics_guided_eval
from forensic_compare.nn_model import build_feature_fusion_model
from scripts import train_physics_guided_net as physics_guided
from scripts.train_physics_guided_net import _standardize_features


def test_feature_fusion_model_forward_shape() -> None:
    model = build_feature_fusion_model(
        "tiny_cnn",
        feature_dim=5,
        num_classes=2,
        hidden_dim=8,
        dropout=0.0,
    )

    logits = model(torch.zeros(2, 3, 32, 32), torch.zeros(2, 5))

    assert logits.shape == (2, 2)


def test_standardize_features_uses_train_statistics() -> None:
    train = np.asarray([[1.0, 2.0], [3.0, 2.0]], dtype=np.float32)
    test = np.asarray([[5.0, 2.0]], dtype=np.float32)

    train_scaled, test_scaled, mean, std = _standardize_features(train, test)

    assert np.allclose(mean, [2.0, 2.0])
    assert np.allclose(std, [1.0, 1.0])
    assert np.allclose(train_scaled[:, 0], [-1.0, 1.0])
    assert np.allclose(test_scaled, [[3.0, 0.0]])


def test_extract_feature_matrix_can_skip_bad_rows(monkeypatch) -> None:
    class DummyDataset:
        def __len__(self) -> int:
            return 3

    def fake_dataset_path(_dataset, index: int) -> str:
        return f"image-{index}.jpg"

    def fake_extract_feature_set(path: str, image_size: int, feature_set: str) -> np.ndarray:
        assert image_size == 32
        assert feature_set == "combined_v3"
        if path == "image-1.jpg":
            raise OSError("bad image")
        return np.asarray([len(path)], dtype=np.float32)

    monkeypatch.setattr(physics_guided, "_dataset_path", fake_dataset_path)
    monkeypatch.setattr(physics_guided, "extract_feature_set", fake_extract_feature_set)

    features, paths, skipped, kept_indices = physics_guided._extract_feature_matrix(
        DummyDataset(),
        feature_set="combined_v3",
        image_size=32,
        desc="test",
        skip_errors=True,
    )

    assert features.shape == (2, 1)
    assert paths == ["image-0.jpg", "image-2.jpg"]
    assert kept_indices == [0, 2]
    assert skipped == [{"path": "image-1.jpg", "error": "OSError('bad image')"}]


def test_standardize_target_features_rejects_dimension_mismatch() -> None:
    target = np.asarray([[1.0, 2.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        physics_guided_eval._standardize_target_features(
            target,
            feature_mean=[1.0],
            feature_std=[1.0],
        )
