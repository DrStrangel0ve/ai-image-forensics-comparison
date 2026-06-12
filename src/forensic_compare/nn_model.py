from __future__ import annotations

import torch
from torch import nn


class TinyCNN(nn.Module):
    """Small CNN used for smoke tests and very fast CPU baselines."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(torch.flatten(x, 1))


class ImageFeatureFusionNet(nn.Module):
    """Fuse an image backbone with physics/signal feature vectors."""

    def __init__(
        self,
        image_backbone: nn.Module,
        image_dim: int,
        feature_dim: int,
        num_classes: int = 2,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.image_backbone = image_backbone
        self.feature_branch = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(image_dim + hidden_dim, num_classes),
        )

    def forward(self, image: torch.Tensor, feature_vector: torch.Tensor) -> torch.Tensor:
        image_embedding = self.image_backbone(image)
        feature_embedding = self.feature_branch(feature_vector)
        return self.classifier(torch.cat([image_embedding, feature_embedding], dim=1))


def _tiny_cnn_backbone() -> tuple[nn.Module, int]:
    model = TinyCNN(num_classes=2)
    return nn.Sequential(model.features, nn.Flatten(1)), 128


def _resnet18_backbone(pretrained: bool) -> tuple[nn.Module, int]:
    from torchvision.models import ResNet18_Weights, resnet18

    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    image_dim = model.fc.in_features
    model.fc = nn.Identity()
    return model, image_dim


def build_model(name: str, num_classes: int = 2, pretrained: bool = False) -> nn.Module:
    normalized = name.lower()
    if normalized in {"tiny", "tiny_cnn", "small_cnn"}:
        return TinyCNN(num_classes=num_classes)
    if normalized == "resnet18":
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    raise ValueError(f"Unsupported model: {name}")


def build_feature_fusion_model(
    image_model: str,
    feature_dim: int,
    num_classes: int = 2,
    pretrained: bool = False,
    hidden_dim: int = 128,
    dropout: float = 0.2,
) -> nn.Module:
    normalized = image_model.lower()
    if normalized in {"tiny", "tiny_cnn", "small_cnn"}:
        image_backbone, image_dim = _tiny_cnn_backbone()
    elif normalized == "resnet18":
        image_backbone, image_dim = _resnet18_backbone(pretrained)
    else:
        raise ValueError(f"Unsupported image model: {image_model}")
    return ImageFeatureFusionNet(
        image_backbone=image_backbone,
        image_dim=image_dim,
        feature_dim=feature_dim,
        num_classes=num_classes,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
