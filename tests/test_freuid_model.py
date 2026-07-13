from __future__ import annotations

import torch

from forensic_compare.freuid_model import build_freuid_model, reverse_gradient, supported_freuid_models


def test_tiny_freuid_multitask_model_shapes() -> None:
    model = build_freuid_model("tiny_cnn", num_types=5, pretrained=False)
    fraud_logits, type_logits = model(torch.randn(3, 3, 64, 64))

    assert fraud_logits.shape == (3,)
    assert type_logits.shape == (3, 5)
    assert "convnext_tiny" in supported_freuid_models()


def test_multiview_freuid_model_shapes_and_gradient_reversal() -> None:
    model = build_freuid_model("tiny_cnn", num_types=3, pretrained=False, multi_view=True)
    fraud_logits, type_logits = model(torch.randn(2, 5, 3, 64, 64), type_adversarial_scale=0.2)

    assert fraud_logits.shape == (2,)
    assert type_logits.shape == (2, 3)

    inputs = torch.tensor([1.0, -2.0], requires_grad=True)
    reverse_gradient(inputs, 0.25).sum().backward()
    assert torch.allclose(inputs.grad, torch.tensor([-0.25, -0.25]))


def test_forensic_residual_adapter_starts_as_identity() -> None:
    model = build_freuid_model(
        "tiny_cnn",
        num_types=2,
        pretrained=False,
        forensic_residual=True,
    )
    adapter = model.encoder[0]
    images = torch.randn(2, 3, 32, 32)

    assert torch.allclose(adapter(images), images)
