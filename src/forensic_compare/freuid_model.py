from __future__ import annotations

import math

import torch
from torch import nn

from .nn_model import TinyCNN


TIMM_FREUID_MODELS = {
    "dinov2_base_518": "vit_base_patch14_dinov2.lvd142m",
    "dinov2_large_518": "vit_large_patch14_dinov2.lvd142m",
    "maxvit_base_512": "maxvit_base_tf_512.in21k_ft_in1k",
    "maxvit_large_512": "maxvit_large_tf_512.in21k_ft_in1k",
}

FREUID_MODEL_INPUT_SIZES = {
    "dinov2_base_518": 518,
    "dinov2_large_518": 518,
    "maxvit_base_512": 512,
    "maxvit_large_512": 512,
}


class LoRALinear(nn.Module):
    """Low-rank update around a frozen linear layer."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.rank
        self.lora_a = nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_b = nn.Linear(self.rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b.weight)
        for parameter in self.base.parameters():
            parameter.requires_grad = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.base(inputs) + self.lora_b(self.lora_a(inputs)) * self.scale


def inject_lora_qkv(module: nn.Module, rank: int, alpha: float) -> tuple[str, ...]:
    """Replace attention QKV projections with trainable low-rank adapters."""

    replaced: list[str] = []

    def visit(parent: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(parent.named_children()):
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child, nn.Linear) and full_name.endswith("attn.qkv"):
                setattr(parent, child_name, LoRALinear(child, rank=rank, alpha=alpha))
                replaced.append(full_name)
            else:
                visit(child, full_name)

    visit(module)
    if not replaced:
        raise ValueError("LoRA requested, but the encoder has no linear attention QKV projections")
    return tuple(replaced)


def configure_encoder_training(
    encoder: nn.Module,
    freeze_encoder: bool = False,
    lora_rank: int = 0,
    lora_alpha: float = 16.0,
) -> tuple[str, ...]:
    """Configure full, frozen, or QKV-LoRA encoder training."""

    if lora_rank < 0:
        raise ValueError("LoRA rank cannot be negative")
    if freeze_encoder or lora_rank > 0:
        for parameter in encoder.parameters():
            parameter.requires_grad = False
    if lora_rank > 0:
        return inject_lora_qkv(encoder, rank=lora_rank, alpha=lora_alpha)
    return ()


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
    """Aggregate multiple document views with attention or mean logits."""

    def __init__(
        self,
        encoder: nn.Module,
        embedding_dim: int,
        num_types: int,
        dropout: float = 0.2,
        view_pooling: str = "attention",
        view_chunk_size: int = 0,
    ) -> None:
        super().__init__()
        if view_pooling not in {"attention", "mean_logits"}:
            raise ValueError(f"Unsupported view pooling: {view_pooling}")
        self.encoder = encoder
        self.view_pooling = view_pooling
        self.view_chunk_size = max(0, int(view_chunk_size))
        self.view_attention = (
            nn.Sequential(
                nn.LayerNorm(embedding_dim),
                nn.Linear(embedding_dim, 1),
            )
            if view_pooling == "attention"
            else None
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
        if self.view_chunk_size > 0 and len(flat_images) > self.view_chunk_size:
            embeddings = torch.cat(
                [
                    self.encoder(flat_images[start : start + self.view_chunk_size])
                    for start in range(0, len(flat_images), self.view_chunk_size)
                ],
                dim=0,
            )
        else:
            embeddings = self.encoder(flat_images)
        if embeddings.ndim > 2:
            embeddings = torch.flatten(embeddings, 1)
        embeddings = embeddings.reshape(batch_size, n_views, -1)
        if self.view_pooling == "attention":
            if self.view_attention is None:
                raise RuntimeError("Attention pooling is not initialized")
            attention = torch.softmax(self.view_attention(embeddings).squeeze(-1), dim=1)
            pooled = torch.sum(embeddings * attention.unsqueeze(-1), dim=1)
            fraud_logits = self.fraud_head(pooled).squeeze(1)
        else:
            pooled = embeddings.mean(dim=1)
            fraud_logits = self.fraud_head(embeddings).squeeze(-1).mean(dim=1)
        type_features = reverse_gradient(pooled, type_adversarial_scale)
        return fraud_logits, self.type_head(type_features)


def supported_freuid_models() -> tuple[str, ...]:
    return (
        "tiny_cnn",
        "efficientnet_b0",
        "convnext_tiny",
        *TIMM_FREUID_MODELS,
    )


def required_freuid_input_size(name: str) -> int | None:
    normalized = name.lower().replace("-", "_")
    return FREUID_MODEL_INPUT_SIZES.get(normalized)


def build_freuid_model(
    name: str,
    num_types: int,
    pretrained: bool = True,
    dropout: float = 0.2,
    multi_view: bool = False,
    forensic_residual: bool = False,
    view_pooling: str = "attention",
    freeze_encoder: bool = False,
    lora_rank: int = 0,
    lora_alpha: float = 16.0,
    view_chunk_size: int = 0,
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
    elif normalized in TIMM_FREUID_MODELS:
        try:
            import timm
        except ImportError as exc:
            raise ImportError("timm is required for MaxViT and DINOv2 FREUID encoders") from exc
        encoder = timm.create_model(
            TIMM_FREUID_MODELS[normalized],
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        embedding_dim = int(encoder.num_features)
    else:
        raise ValueError(f"Unsupported FREUID model: {name}")
    configure_encoder_training(
        encoder,
        freeze_encoder=freeze_encoder,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )
    if forensic_residual:
        encoder = nn.Sequential(ForensicResidualAdapter(), encoder)
    if multi_view:
        return FreuidMultiViewModel(
            encoder=encoder,
            embedding_dim=embedding_dim,
            num_types=num_types,
            dropout=dropout,
            view_pooling=view_pooling,
            view_chunk_size=view_chunk_size,
        )
    return FreuidMultiTaskModel(
        encoder=encoder,
        embedding_dim=embedding_dim,
        num_types=num_types,
        dropout=dropout,
    )
