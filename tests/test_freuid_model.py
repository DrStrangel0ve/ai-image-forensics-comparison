from __future__ import annotations

import sys
from types import SimpleNamespace

import torch
from torch import nn

from forensic_compare.freuid_model import (
    LoRALinear,
    build_freuid_model,
    configure_encoder_training,
    required_freuid_input_size,
    reverse_gradient,
    supported_freuid_models,
)


def test_tiny_freuid_multitask_model_shapes() -> None:
    model = build_freuid_model("tiny_cnn", num_types=5, pretrained=False)
    fraud_logits, type_logits = model(torch.randn(3, 3, 64, 64))

    assert fraud_logits.shape == (3,)
    assert type_logits.shape == (3, 5)
    assert "convnext_tiny" in supported_freuid_models()
    assert "dinov2_large_518" in supported_freuid_models()
    assert "maxvit_base_512" in supported_freuid_models()
    assert required_freuid_input_size("dinov2-large-518") == 518
    assert required_freuid_input_size("tiny_cnn") is None


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


def test_mean_logit_multiview_pooling_and_chunking() -> None:
    model = build_freuid_model(
        "tiny_cnn",
        num_types=3,
        pretrained=False,
        multi_view=True,
        view_pooling="mean_logits",
        view_chunk_size=2,
    )

    fraud_logits, type_logits = model(torch.randn(2, 5, 3, 64, 64))

    assert fraud_logits.shape == (2,)
    assert type_logits.shape == (2, 3)


def test_qkv_lora_freezes_base_and_trains_only_adapters() -> None:
    class Attention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.qkv = nn.Linear(8, 24)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.qkv(inputs)

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attn = Attention()

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.attn(inputs)

    encoder = nn.Sequential(Block())
    replaced = configure_encoder_training(encoder, lora_rank=2, lora_alpha=4.0)

    assert replaced == ("0.attn.qkv",)
    assert isinstance(encoder[0].attn.qkv, LoRALinear)
    trainable = [name for name, parameter in encoder.named_parameters() if parameter.requires_grad]
    assert trainable == ["0.attn.qkv.lora_a.weight", "0.attn.qkv.lora_b.weight"]
    assert encoder(torch.randn(3, 8)).shape == (3, 24)


def test_dinov2_uses_checkpoint_compatible_token_pooling(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    class FakeEncoder(nn.Module):
        num_features = 8

        def __init__(self) -> None:
            super().__init__()
            self.projection = nn.Linear(8, 8)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.projection(inputs)

    def fake_create_model(name: str, **kwargs: object) -> FakeEncoder:
        calls.append({"name": name, **kwargs})
        return FakeEncoder()

    monkeypatch.setitem(sys.modules, "timm", SimpleNamespace(create_model=fake_create_model))

    build_freuid_model("dinov2_base_518", num_types=2, pretrained=True, freeze_encoder=True)

    assert calls == [
        {
            "name": "vit_base_patch14_dinov2.lvd142m",
            "pretrained": True,
            "num_classes": 0,
            "global_pool": "token",
        }
    ]
