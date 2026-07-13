from __future__ import annotations

import io
import random

import torch
from PIL import Image, ImageOps
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Letterbox:
    def __init__(self, size: int, fill: tuple[int, int, int] = (127, 127, 127)) -> None:
        self.size = int(size)
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        return ImageOps.pad(
            image,
            (self.size, self.size),
            method=Image.Resampling.BICUBIC,
            color=self.fill,
            centering=(0.5, 0.5),
        )


class RandomJpeg:
    def __init__(self, probability: float, quality_range: tuple[int, int] = (40, 96)) -> None:
        self.probability = float(probability)
        self.quality_range = quality_range

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.probability:
            return image
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=random.randint(*self.quality_range))
        buffer.seek(0)
        with Image.open(buffer) as encoded:
            return encoded.convert("RGB").copy()


class RandomDownsample:
    def __init__(self, probability: float, scale_range: tuple[float, float] = (0.35, 0.85)) -> None:
        self.probability = float(probability)
        self.scale_range = scale_range

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.probability:
            return image
        scale = random.uniform(*self.scale_range)
        reduced = image.resize(
            (max(16, round(image.width * scale)), max(16, round(image.height * scale))),
            Image.Resampling.BILINEAR,
        )
        return reduced.resize(image.size, Image.Resampling.BICUBIC)


class RandomTensorNoise:
    def __init__(self, probability: float, sigma_range: tuple[float, float] = (0.002, 0.018)) -> None:
        self.probability = float(probability)
        self.sigma_range = sigma_range

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.probability:
            return tensor
        sigma = random.uniform(*self.sigma_range)
        return torch.clamp(tensor + torch.randn_like(tensor) * sigma, 0.0, 1.0)


class DocumentViewTransform:
    """Create one global view plus overlapping full-resolution grid tiles."""

    def __init__(
        self,
        size: int,
        grid_rows: int = 0,
        grid_cols: int = 0,
        overlap: float = 0.08,
        shared_transform=None,
        tensor_transform=None,
    ) -> None:
        self.size = int(size)
        self.grid_rows = max(0, int(grid_rows))
        self.grid_cols = max(0, int(grid_cols))
        self.overlap = max(0.0, min(float(overlap), 0.45))
        self.shared_transform = shared_transform
        self.tensor_transform = tensor_transform or transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
        self.letterbox = Letterbox(self.size)

    def _tiles(self, image: Image.Image) -> list[Image.Image]:
        if self.grid_rows <= 0 or self.grid_cols <= 0:
            return []
        tiles: list[Image.Image] = []
        cell_width = image.width / self.grid_cols
        cell_height = image.height / self.grid_rows
        x_margin = cell_width * self.overlap
        y_margin = cell_height * self.overlap
        for row in range(self.grid_rows):
            for column in range(self.grid_cols):
                left = max(0, round(column * cell_width - x_margin))
                top = max(0, round(row * cell_height - y_margin))
                right = min(image.width, round((column + 1) * cell_width + x_margin))
                bottom = min(image.height, round((row + 1) * cell_height + y_margin))
                tiles.append(image.crop((left, top, right, bottom)))
        return tiles

    def __call__(self, image: Image.Image) -> torch.Tensor:
        image = image.convert("RGB")
        if self.shared_transform is not None:
            image = self.shared_transform(image)
        views = [image, *self._tiles(image)]
        tensors = [self.tensor_transform(self.letterbox(view)) for view in views]
        if len(tensors) == 1:
            return tensors[0]
        return torch.stack(tensors, dim=0)


def build_document_transforms(
    image_size: int,
    grid_rows: int = 0,
    grid_cols: int = 0,
    jpeg_probability: float = 0.25,
    capture_strength: float = 0.0,
) -> tuple[DocumentViewTransform, DocumentViewTransform]:
    strength = max(0.0, min(float(capture_strength), 1.0))
    train_shared = transforms.Compose(
        [
            RandomJpeg(jpeg_probability, quality_range=(35, 96)),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.08 + 0.22 * strength,
                        contrast=0.08 + 0.22 * strength,
                        saturation=0.05 + 0.15 * strength,
                        hue=0.01 + 0.02 * strength,
                    )
                ],
                p=0.35 + 0.4 * strength,
            ),
            transforms.RandomPerspective(
                distortion_scale=0.12 * strength,
                p=0.35 * strength,
                fill=127,
            ),
            transforms.RandomAffine(
                degrees=1.5 + 2.5 * strength,
                translate=(0.012 + 0.025 * strength, 0.012 + 0.025 * strength),
                scale=(0.98 - 0.08 * strength, 1.02 + 0.04 * strength),
                fill=127,
            ),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 0.3 + 1.4 * strength))],
                p=0.3 * strength,
            ),
            RandomDownsample(probability=0.35 * strength),
        ]
    )
    tensor_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            RandomTensorNoise(probability=0.3 * strength),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    train = DocumentViewTransform(
        image_size,
        grid_rows=grid_rows,
        grid_cols=grid_cols,
        shared_transform=train_shared,
        tensor_transform=tensor_transform,
    )
    evaluate = DocumentViewTransform(image_size, grid_rows=grid_rows, grid_cols=grid_cols)
    return train, evaluate
