from __future__ import annotations

import hashlib
from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter

ROBUSTNESS_VARIANTS = (
    "jpeg70",
    "jpeg50",
    "jpeg30",
    "blur1",
    "resize_half",
    "crop85",
    "noise3",
    "screenshot",
    "social_square",
)


def jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as jpeg:
        return jpeg.convert("RGB")


def center_crop_resize(image: Image.Image, fraction: float) -> Image.Image:
    width, height = image.size
    crop_width = max(1, int(round(width * fraction)))
    crop_height = max(1, int(round(height * fraction)))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    cropped = image.crop((left, top, left + crop_width, top + crop_height))
    return cropped.resize((width, height), Image.Resampling.BICUBIC)


def center_square_crop_resize(image: Image.Image) -> Image.Image:
    width, height = image.size
    side = max(1, min(width, height))
    left = (width - side) // 2
    top = (height - side) // 2
    cropped = image.crop((left, top, left + side, top + side))
    return cropped.resize((width, height), Image.Resampling.BICUBIC)


def resize_half_roundtrip(image: Image.Image) -> Image.Image:
    width, height = image.size
    down = image.resize((max(1, width // 2), max(1, height // 2)), Image.Resampling.BICUBIC)
    return down.resize((width, height), Image.Resampling.BICUBIC)


def deterministic_gaussian_noise(image: Image.Image, sigma: float) -> Image.Image:
    seed = int.from_bytes(hashlib.blake2b(image.tobytes(), digest_size=8).digest(), "little")
    rng = np.random.default_rng(seed)
    array = np.asarray(image).astype(np.float32)
    noisy = array + rng.normal(loc=0.0, scale=sigma, size=array.shape)
    return Image.fromarray(np.clip(noisy, 0, 255).astype(np.uint8), mode="RGB")


def screenshot_roundtrip(image: Image.Image) -> Image.Image:
    width, height = image.size
    down_size = (max(1, int(round(width * 0.75))), max(1, int(round(height * 0.75))))
    down = image.resize(down_size, Image.Resampling.BILINEAR)
    restored = down.resize((width, height), Image.Resampling.BILINEAR)
    return jpeg_roundtrip(restored, quality=85)


def social_square_roundtrip(image: Image.Image) -> Image.Image:
    return jpeg_roundtrip(center_square_crop_resize(image), quality=80)


def apply_robustness_variant(image: Image.Image, variant: str) -> Image.Image:
    image = image.convert("RGB")
    if variant == "jpeg70":
        return jpeg_roundtrip(image, quality=70)
    if variant == "jpeg50":
        return jpeg_roundtrip(image, quality=50)
    if variant == "jpeg30":
        return jpeg_roundtrip(image, quality=30)
    if variant == "blur1":
        return image.filter(ImageFilter.GaussianBlur(radius=1.0))
    if variant == "resize_half":
        return resize_half_roundtrip(image)
    if variant == "crop85":
        return center_crop_resize(image, fraction=0.85)
    if variant == "noise3":
        return deterministic_gaussian_noise(image, sigma=3.0)
    if variant == "screenshot":
        return screenshot_roundtrip(image)
    if variant == "social_square":
        return social_square_roundtrip(image)
    raise ValueError(f"Unsupported variant: {variant}")
