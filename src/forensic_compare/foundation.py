from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torchvision import transforms

from .nn_model import TinyCNN


@dataclass(frozen=True)
class FrozenEncoderSpec:
    model: nn.Module
    embedding_dim: int
    image_size: int
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    weights: str | None


HF_ENCODER_CONFIGS: dict[str, dict[str, Any]] = {
    "clip_vit_b_32": {
        "model_id": "openai/clip-vit-base-patch32",
        "loader": "clip_vision",
        "image_size": 224,
        "mean": (0.48145466, 0.4578275, 0.40821073),
        "std": (0.26862954, 0.26130258, 0.27577711),
    },
    "dinov2_vits14": {
        "model_id": "facebook/dinov2-small",
        "loader": "auto",
        "image_size": 224,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "interpolate_pos_encoding": True,
    },
    "dinov2_vitb14": {
        "model_id": "facebook/dinov2-base",
        "loader": "auto",
        "image_size": 224,
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225),
        "interpolate_pos_encoding": True,
    },
}


class HFVisionEncoder(nn.Module):
    def __init__(self, model: nn.Module, interpolate_pos_encoding: bool = False) -> None:
        super().__init__()
        self.model = model
        self.interpolate_pos_encoding = interpolate_pos_encoding

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        kwargs = {"pixel_values": images}
        if self.interpolate_pos_encoding:
            kwargs["interpolate_pos_encoding"] = True
        outputs = self.model(**kwargs)
        pooler_output = getattr(outputs, "pooler_output", None)
        if pooler_output is not None:
            return pooler_output
        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        if last_hidden_state is None:
            raise RuntimeError("Hugging Face vision model did not return hidden states")
        return last_hidden_state[:, 0]


def _imagenet_norm() -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def supported_frozen_encoders() -> tuple[str, ...]:
    return (
        "tiny_cnn",
        "resnet18",
        "resnet50",
        "convnext_tiny",
        "efficientnet_b0",
        "vit_b_16",
        "swin_t",
        *HF_ENCODER_CONFIGS.keys(),
    )


def _build_hf_encoder(normalized: str, pretrained: bool) -> FrozenEncoderSpec:
    if not pretrained:
        raise ValueError(
            f"{normalized} is a foundation encoder and requires --pretrained to load public weights"
        )
    try:
        from transformers import AutoModel, CLIPVisionModel
    except ImportError as exc:
        raise ImportError(
            "CLIP/DINO frozen encoders require the optional 'transformers' dependency. "
            "Install the project requirements again or run `python -m pip install transformers`."
        ) from exc

    config = HF_ENCODER_CONFIGS[normalized]
    model_id = str(config["model_id"])
    if config["loader"] == "clip_vision":
        model = CLIPVisionModel.from_pretrained(model_id)
    else:
        model = AutoModel.from_pretrained(model_id)
    hidden_size = getattr(model.config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError(f"Could not infer embedding dimension for {model_id}")
    encoder = HFVisionEncoder(
        model,
        interpolate_pos_encoding=bool(config.get("interpolate_pos_encoding", False)),
    )
    for parameter in encoder.parameters():
        parameter.requires_grad = False
    encoder.eval()
    return FrozenEncoderSpec(
        model=encoder,
        embedding_dim=int(hidden_size),
        image_size=int(config["image_size"]),
        mean=tuple(config["mean"]),
        std=tuple(config["std"]),
        weights=model_id,
    )


def build_frozen_encoder(name: str, pretrained: bool = True) -> FrozenEncoderSpec:
    normalized = name.lower().replace("-", "_")
    mean, std = _imagenet_norm()
    weights_name: str | None = None

    if normalized in {"tiny", "tiny_cnn"}:
        model = TinyCNN(num_classes=2)
        encoder = nn.Sequential(model.features, nn.Flatten(1))
        for parameter in encoder.parameters():
            parameter.requires_grad = False
        encoder.eval()
        return FrozenEncoderSpec(
            model=encoder,
            embedding_dim=128,
            image_size=64,
            mean=mean,
            std=std,
            weights=None,
        )

    if normalized == "resnet18":
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        embedding_dim = model.fc.in_features
        model.fc = nn.Identity()
        weights_name = str(weights) if weights else None
        image_size = 224
    elif normalized == "resnet50":
        from torchvision.models import ResNet50_Weights, resnet50

        weights = ResNet50_Weights.DEFAULT if pretrained else None
        model = resnet50(weights=weights)
        embedding_dim = model.fc.in_features
        model.fc = nn.Identity()
        weights_name = str(weights) if weights else None
        image_size = 224
    elif normalized == "convnext_tiny":
        from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = convnext_tiny(weights=weights)
        embedding_dim = model.classifier[2].in_features
        model.classifier[2] = nn.Identity()
        weights_name = str(weights) if weights else None
        image_size = 224
    elif normalized == "efficientnet_b0":
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        embedding_dim = model.classifier[1].in_features
        model.classifier[1] = nn.Identity()
        weights_name = str(weights) if weights else None
        image_size = 224
    elif normalized == "vit_b_16":
        from torchvision.models import ViT_B_16_Weights, vit_b_16

        weights = ViT_B_16_Weights.DEFAULT if pretrained else None
        model = vit_b_16(weights=weights)
        embedding_dim = model.heads.head.in_features
        model.heads.head = nn.Identity()
        weights_name = str(weights) if weights else None
        image_size = 224
    elif normalized == "swin_t":
        from torchvision.models import Swin_T_Weights, swin_t

        weights = Swin_T_Weights.DEFAULT if pretrained else None
        model = swin_t(weights=weights)
        embedding_dim = model.head.in_features
        model.head = nn.Identity()
        weights_name = str(weights) if weights else None
        image_size = 224
    elif normalized in HF_ENCODER_CONFIGS:
        return _build_hf_encoder(normalized, pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported frozen encoder: {name}")

    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return FrozenEncoderSpec(
        model=model,
        embedding_dim=embedding_dim,
        image_size=image_size,
        mean=mean,
        std=std,
        weights=weights_name,
    )


def frozen_encoder_transform(
    image_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


@torch.no_grad()
def encode_batch(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    embeddings = model(images)
    if embeddings.ndim > 2:
        embeddings = torch.flatten(embeddings, 1)
    return embeddings
