from __future__ import annotations

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from .photometric import FEATURE_NAMES as PHOTOMETRIC_FEATURE_NAMES
from .photometric import extract_features as extract_photometric_features

NOISE_FEATURE_NAMES = [
    "noise_gray_std",
    "noise_abs_mean",
    "noise_abs_p90",
    "noise_abs_p99",
    "noise_entropy",
    "ela_abs_mean",
    "ela_abs_std",
    "ela_abs_p95",
    "jpeg_block_col_ratio",
    "jpeg_block_row_ratio",
    "fft_low_ratio",
    "fft_mid_ratio",
    "fft_high_ratio",
    "fft_centroid",
    "fft_axis_anisotropy",
    "gradient_orientation_entropy",
    "chroma_u_std",
    "chroma_v_std",
    "chroma_noise_corr",
]


def _load_rgb(path: str | Path, image_size: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32) / 255.0


def _gray(rgb: np.ndarray) -> np.ndarray:
    return (
        0.2126 * rgb[:, :, 0]
        + 0.7152 * rgb[:, :, 1]
        + 0.0722 * rgb[:, :, 2]
    ).astype(np.float32)


def _box_blur(values: np.ndarray, radius: int = 1) -> np.ndarray:
    padded = np.pad(values, radius, mode="reflect")
    acc = np.zeros_like(values)
    width = 2 * radius + 1
    for dy in range(width):
        for dx in range(width):
            acc += padded[dy : dy + values.shape[0], dx : dx + values.shape[1]]
    return acc / float(width * width)


def _entropy(values: np.ndarray, bins: int = 32) -> float:
    flat = values.reshape(-1)
    if float(flat.max() - flat.min()) < 1e-8:
        return 0.0
    hist, _ = np.histogram(flat, bins=bins, density=False)
    probs = hist.astype(np.float64) / max(1, hist.sum())
    return -float(np.sum(probs * np.log2(np.clip(probs, 1e-12, None))))


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    if float(a_flat.std()) < 1e-8 or float(b_flat.std()) < 1e-8:
        return 0.0
    return float(np.corrcoef(a_flat, b_flat)[0, 1])


def _ela_diff(rgb: np.ndarray) -> np.ndarray:
    image = Image.fromarray(np.clip(rgb * 255, 0, 255).astype(np.uint8))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=90)
    buffer.seek(0)
    with Image.open(buffer) as jpeg:
        jpeg_rgb = np.asarray(jpeg.convert("RGB"), dtype=np.float32) / 255.0
    return np.abs(rgb - jpeg_rgb)


def _block_boundary_ratio(values: np.ndarray, axis: int) -> float:
    diffs = np.abs(np.diff(values, axis=axis))
    length = diffs.shape[axis]
    if length < 9:
        return 0.0
    indices = np.arange(length)
    boundary_mask = ((indices + 1) % 8) == 0
    if axis == 0:
        boundary = diffs[boundary_mask, :]
        non_boundary = diffs[~boundary_mask, :]
    else:
        boundary = diffs[:, boundary_mask]
        non_boundary = diffs[:, ~boundary_mask]
    return float(boundary.mean() / np.clip(non_boundary.mean(), 1e-8, None))


def _fft_features(gray: np.ndarray) -> dict[str, float]:
    centered = gray - gray.mean()
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(centered))) ** 2
    height, width = gray.shape
    yy, xx = np.mgrid[:height, :width]
    cy = (height - 1) / 2.0
    cx = (width - 1) / 2.0
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    radius_norm = radius / np.clip(radius.max(), 1e-8, None)
    total = float(spectrum.sum()) + 1e-8
    low = spectrum[radius_norm < 0.15].sum() / total
    mid = spectrum[(radius_norm >= 0.15) & (radius_norm < 0.45)].sum() / total
    high = spectrum[radius_norm >= 0.45].sum() / total
    vertical_axis = spectrum[:, max(0, int(cx) - 1) : int(cx) + 2].sum()
    horizontal_axis = spectrum[max(0, int(cy) - 1) : int(cy) + 2, :].sum()
    anisotropy = abs(float(vertical_axis - horizontal_axis)) / total
    return {
        "fft_low_ratio": float(low),
        "fft_mid_ratio": float(mid),
        "fft_high_ratio": float(high),
        "fft_centroid": float((spectrum * radius_norm).sum() / total),
        "fft_axis_anisotropy": anisotropy,
    }


def _gradient_orientation_entropy(gray: np.ndarray) -> float:
    gy, gx = np.gradient(gray)
    magnitude = np.sqrt(gx**2 + gy**2)
    angle = np.arctan2(gy, gx)
    hist, _ = np.histogram(angle, bins=18, range=(-np.pi, np.pi), weights=magnitude)
    probs = hist.astype(np.float64) / max(float(hist.sum()), 1e-8)
    return -float(np.sum(probs * np.log2(np.clip(probs, 1e-12, None))))


def extract_noise_features(path: str | Path, image_size: int = 128) -> np.ndarray:
    rgb = _load_rgb(path, image_size)
    gray = _gray(rgb)
    residual = gray - _box_blur(gray, radius=1)
    residual_abs = np.abs(residual)
    ela = _ela_diff(rgb)
    ela_abs = np.mean(ela, axis=2)
    chroma_u = rgb[:, :, 0] - gray
    chroma_v = rgb[:, :, 2] - gray

    values = {
        "noise_gray_std": float(gray.std()),
        "noise_abs_mean": float(residual_abs.mean()),
        "noise_abs_p90": float(np.percentile(residual_abs, 90)),
        "noise_abs_p99": float(np.percentile(residual_abs, 99)),
        "noise_entropy": _entropy(residual, bins=48),
        "ela_abs_mean": float(ela_abs.mean()),
        "ela_abs_std": float(ela_abs.std()),
        "ela_abs_p95": float(np.percentile(ela_abs, 95)),
        "jpeg_block_col_ratio": _block_boundary_ratio(gray, axis=1),
        "jpeg_block_row_ratio": _block_boundary_ratio(gray, axis=0),
        "gradient_orientation_entropy": _gradient_orientation_entropy(gray),
        "chroma_u_std": float(chroma_u.std()),
        "chroma_v_std": float(chroma_v.std()),
        "chroma_noise_corr": _safe_corr(np.sqrt(chroma_u**2 + chroma_v**2), residual_abs),
    }
    values.update(_fft_features(gray))
    return np.asarray([values[name] for name in NOISE_FEATURE_NAMES], dtype=np.float32)


def feature_names(feature_set: str) -> list[str]:
    if feature_set == "photometric":
        return list(PHOTOMETRIC_FEATURE_NAMES)
    if feature_set == "noise":
        return list(NOISE_FEATURE_NAMES)
    if feature_set == "combined":
        return list(PHOTOMETRIC_FEATURE_NAMES) + list(NOISE_FEATURE_NAMES)
    raise ValueError(f"Unsupported feature set: {feature_set}")


def extract_feature_set(path: str | Path, image_size: int, feature_set: str) -> np.ndarray:
    if feature_set == "photometric":
        return extract_photometric_features(path, image_size=image_size)
    if feature_set == "noise":
        return extract_noise_features(path, image_size=image_size)
    if feature_set == "combined":
        return np.concatenate(
            [
                extract_photometric_features(path, image_size=image_size),
                extract_noise_features(path, image_size=image_size),
            ]
        )
    raise ValueError(f"Unsupported feature set: {feature_set}")
