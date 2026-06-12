from __future__ import annotations

from dataclasses import dataclass

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


def _imagenet_norm() -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


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
