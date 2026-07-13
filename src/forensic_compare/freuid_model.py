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

    def forward(
        self,
        images: torch.Tensor,
        type_adversarial_scale: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        embeddings = self.encoder(images)
        if embeddings.ndim > 2:
            embeddings = torch.flatten(embeddings, 1)
        type_features = reverse_gradient(embeddings, type_adversarial_scale)
        return self.fraud_head(embeddings).squeeze(1), self.type_head(type_features)


class _GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return inputs.view_as(inputs)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.scale * grad_output, None


def reverse_gradient(inputs: torch.Tensor, scale: float) -> torch.Tensor:
    return _GradientReverse.apply(inputs, float(scale))


class ForensicResidualAdapter(nn.Module):
    """Expose a trainable RGB plus high-pass residual mixture to a pretrained encoder."""

    def __init__(self) -> None:
        super().__init__()
        self.mix = nn.Conv2d(6, 3, kernel_size=1, bias=False)
        with torch.no_grad():
            self.mix.weight.zero_()
            for channel in range(3):
                self.mix.weight[channel, channel, 0, 0] = 1.0

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        local_mean = torch.nn.functional.avg_pool2d(images, kernel_size=5, stride=1, padding=2)
        residual = images - local_mean
        return self.mix(torch.cat([images, residual], dim=1))


class FreuidMultiViewModel(nn.Module):
    """Aggregate a global view and local document tiles with learned attention."""

    def __init__(self, encoder: nn.Module, embedding_dim: int, num_types: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.encoder = encoder
        self.view_attention = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, 1),
        )
        self.fraud_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(embedding_dim, 1))
        self.type_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(embedding_dim, num_types))

    def forward(
        self,
        images: torch.Tensor,
        type_adversarial_scale: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if images.ndim == 4:
            images = images.unsqueeze(1)
        if images.ndim != 5:
            raise ValueError(f"Expected BCHW or BVCHW input, got shape {tuple(images.shape)}")
        batch_size, n_views, channels, height, width = images.shape
        flat_images = images.reshape(batch_size * n_views, channels, height, width)
        embeddings = self.encoder(flat_images)
        if embeddings.ndim > 2:
            embeddings = torch.flatten(embeddings, 1)
        embeddings = embeddings.reshape(batch_size, n_views, -1)
        attention = torch.softmax(self.view_attention(embeddings).squeeze(-1), dim=1)
        pooled = torch.sum(embeddings * attention.unsqueeze(-1), dim=1)
        fraud_logits = self.fraud_head(pooled).squeeze(1)
        type_features = reverse_gradient(pooled, type_adversarial_scale)
        return fraud_logits, self.type_head(type_features)


def supported_freuid_models() -> tuple[str, ...]:
    return ("tiny_cnn", "efficientnet_b0", "convnext_tiny")


def build_freuid_model(
    name: str,
    num_types: int,
    pretrained: bool = True,
    dropout: float = 0.2,
    multi_view: bool = False,
    forensic_residual: bool = False,
) -> FreuidMultiTaskModel | FreuidMultiViewModel:
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
    if forensic_residual:
        encoder = nn.Sequential(ForensicResidualAdapter(), encoder)
    model_class = FreuidMultiViewModel if multi_view else FreuidMultiTaskModel
    return model_class(
        encoder=encoder,
        embedding_dim=embedding_dim,
        num_types=num_types,
        dropout=dropout,
    )
