from __future__ import annotations

import torch
from torch import nn

from .nn_model import TinyCNN


class FreuidMultiTaskModel(nn.Module):
    """Shared image encoder with fraud and document-type heads."""

    def __init__(self, encoder: nn.Module, embedding_dim: int, num_types: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.encoder = encoder
        self.fraud_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(embedding_dim, 1))
        self.type_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(embedding_dim, num_types))

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.encoder(images)
        if embeddings.ndim > 2:
            embeddings = torch.flatten(embeddings, 1)
        return self.fraud_head(embeddings).squeeze(1), self.type_head(embeddings)


def supported_freuid_models() -> tuple[str, ...]:
    return ("tiny_cnn", "efficientnet_b0", "convnext_tiny")


def build_freuid_model(
    name: str,
    num_types: int,
    pretrained: bool = True,
    dropout: float = 0.2,
) -> FreuidMultiTaskModel:
    normalized = name.lower().replace("-", "_")
    if normalized in {"tiny", "tiny_cnn"}:
        tiny = TinyCNN(num_classes=2)
        encoder = nn.Sequential(tiny.features, nn.Flatten(1))
        embedding_dim = 128
    elif normalized == "efficientnet_b0":
        from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        embedding_dim = int(model.classifier[1].in_features)
        encoder = nn.Sequential(model.features, model.avgpool, nn.Flatten(1))
    elif normalized == "convnext_tiny":
        from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny

        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = convnext_tiny(weights=weights)
        embedding_dim = int(model.classifier[2].in_features)
        encoder = nn.Sequential(
            model.features,
            model.avgpool,
            model.classifier[0],
            model.classifier[1],
        )
    else:
        raise ValueError(f"Unsupported FREUID model: {name}")
    return FreuidMultiTaskModel(
        encoder=encoder,
        embedding_dim=embedding_dim,
        num_types=num_types,
        dropout=dropout,
    )
