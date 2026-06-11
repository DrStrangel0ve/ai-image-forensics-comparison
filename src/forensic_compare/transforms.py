from __future__ import annotations

from io import BytesIO

from PIL import Image, ImageFilter

ROBUSTNESS_VARIANTS = ("jpeg70", "blur1", "resize_half", "crop85")


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


def resize_half_roundtrip(image: Image.Image) -> Image.Image:
    width, height = image.size
    down = image.resize((max(1, width // 2), max(1, height // 2)), Image.Resampling.BICUBIC)
    return down.resize((width, height), Image.Resampling.BICUBIC)


def apply_robustness_variant(image: Image.Image, variant: str) -> Image.Image:
    image = image.convert("RGB")
    if variant == "jpeg70":
        return jpeg_roundtrip(image, quality=70)
    if variant == "blur1":
        return image.filter(ImageFilter.GaussianBlur(radius=1.0))
    if variant == "resize_half":
        return resize_half_roundtrip(image)
    if variant == "crop85":
        return center_crop_resize(image, fraction=0.85)
    raise ValueError(f"Unsupported variant: {variant}")
