from __future__ import annotations

import torch

from forensic_compare.freuid_model import build_freuid_model, supported_freuid_models


def test_tiny_freuid_multitask_model_shapes() -> None:
    model = build_freuid_model("tiny_cnn", num_types=5, pretrained=False)
    fraud_logits, type_logits = model(torch.randn(3, 3, 64, 64))

    assert fraud_logits.shape == (3,)
    assert type_logits.shape == (3, 5)
    assert "convnext_tiny" in supported_freuid_models()
