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

NOISE_V2_EXTRA_FEATURE_NAMES = [
    "noise_laplacian_abs_mean",
    "noise_laplacian_abs_p95",
    "noise_laplacian_entropy",
    "noise_multiscale_abs_mean",
    "noise_multiscale_abs_p95",
    "noise_multiscale_entropy",
    "noise_tile8_entropy_mean",
    "noise_tile8_entropy_std",
    "noise_tile8_entropy_p90",
    "noise_neighbor_corr_x",
    "noise_neighbor_corr_y",
]

NOISE_V2_FEATURE_NAMES = NOISE_FEATURE_NAMES + NOISE_V2_EXTRA_FEATURE_NAMES

NOISE_V3_EXTRA_FEATURE_NAMES = [
    "jpeg_q70_abs_mean",
    "jpeg_q70_abs_p95",
    "jpeg_q95_abs_mean",
    "jpeg_q95_abs_p95",
    "jpeg_q70_q95_mean_ratio",
    "jpeg_q90_q95_mean_delta",
    "residual_phase8_std",
    "residual_phase8_peak_ratio",
    "residual_phase8_contrast",
    "ela_phase8_std",
    "ela_phase8_peak_ratio",
    "ela_phase8_contrast",
    "residual_rg_corr",
    "residual_gb_corr",
    "residual_rb_corr",
    "residual_rgb_abs_std_ratio",
    "residual_chroma_luma_abs_ratio",
    "residual_tile16_std_mean",
    "residual_tile16_std_std",
    "residual_tile16_std_p90",
]

NOISE_V3_FEATURE_NAMES = NOISE_V2_FEATURE_NAMES + NOISE_V3_EXTRA_FEATURE_NAMES


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


def _jpeg_reencode_diff(rgb: np.ndarray, quality: int) -> np.ndarray:
    image = Image.fromarray(np.clip(rgb * 255, 0, 255).astype(np.uint8))
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as jpeg:
        jpeg_rgb = np.asarray(jpeg.convert("RGB"), dtype=np.float32) / 255.0
    return np.abs(rgb - jpeg_rgb)


def _ela_diff(rgb: np.ndarray) -> np.ndarray:
    return _jpeg_reencode_diff(rgb, quality=90)


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


def _laplacian(values: np.ndarray) -> np.ndarray:
    padded = np.pad(values, 1, mode="reflect")
    center = padded[1:-1, 1:-1]
    return (
        -4.0 * center
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    )


def _tile_entropy_stats(values: np.ndarray, tile_size: int = 8) -> dict[str, float]:
    entropies = []
    height, width = values.shape
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            tile = values[y : y + tile_size, x : x + tile_size]
            if tile.shape[0] >= 4 and tile.shape[1] >= 4:
                entropies.append(_entropy(tile, bins=16))
    if not entropies:
        return {
            "noise_tile8_entropy_mean": 0.0,
            "noise_tile8_entropy_std": 0.0,
            "noise_tile8_entropy_p90": 0.0,
        }
    values_array = np.asarray(entropies, dtype=np.float32)
    return {
        "noise_tile8_entropy_mean": float(values_array.mean()),
        "noise_tile8_entropy_std": float(values_array.std()),
        "noise_tile8_entropy_p90": float(np.percentile(values_array, 90)),
    }


def _noise_v2_extra_features(gray: np.ndarray, residual: np.ndarray) -> dict[str, float]:
    laplacian = _laplacian(gray)
    laplacian_abs = np.abs(laplacian)
    multiscale = gray - _box_blur(gray, radius=3)
    multiscale_abs = np.abs(multiscale)
    values = {
        "noise_laplacian_abs_mean": float(laplacian_abs.mean()),
        "noise_laplacian_abs_p95": float(np.percentile(laplacian_abs, 95)),
        "noise_laplacian_entropy": _entropy(laplacian, bins=48),
        "noise_multiscale_abs_mean": float(multiscale_abs.mean()),
        "noise_multiscale_abs_p95": float(np.percentile(multiscale_abs, 95)),
        "noise_multiscale_entropy": _entropy(multiscale, bins=48),
        "noise_neighbor_corr_x": _safe_corr(residual[:, :-1], residual[:, 1:]),
        "noise_neighbor_corr_y": _safe_corr(residual[:-1, :], residual[1:, :]),
    }
    values.update(_tile_entropy_stats(np.abs(residual), tile_size=8))
    return values


def _phase8_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
    phase_means = []
    for y in range(8):
        for x in range(8):
            phase = values[y::8, x::8]
            if phase.size:
                phase_means.append(float(phase.mean()))
    if not phase_means:
        return {
            f"{prefix}_phase8_std": 0.0,
            f"{prefix}_phase8_peak_ratio": 0.0,
            f"{prefix}_phase8_contrast": 0.0,
        }
    phase_values = np.asarray(phase_means, dtype=np.float32)
    mean_value = float(phase_values.mean())
    return {
        f"{prefix}_phase8_std": float(phase_values.std()),
        f"{prefix}_phase8_peak_ratio": float(phase_values.max() / np.clip(mean_value, 1e-8, None)),
        f"{prefix}_phase8_contrast": float(
            (phase_values.max() - phase_values.min()) / np.clip(mean_value, 1e-8, None)
        ),
    }


def _tile_std_stats(values: np.ndarray, tile_size: int, prefix: str) -> dict[str, float]:
    stds = []
    height, width = values.shape
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            tile = values[y : y + tile_size, x : x + tile_size]
            if tile.shape[0] >= 4 and tile.shape[1] >= 4:
                stds.append(float(tile.std()))
    if not stds:
        return {
            f"{prefix}_std_mean": 0.0,
            f"{prefix}_std_std": 0.0,
            f"{prefix}_std_p90": 0.0,
        }
    values_array = np.asarray(stds, dtype=np.float32)
    return {
        f"{prefix}_std_mean": float(values_array.mean()),
        f"{prefix}_std_std": float(values_array.std()),
        f"{prefix}_std_p90": float(np.percentile(values_array, 90)),
    }


def _rgb_residual_features(rgb: np.ndarray, gray_residual: np.ndarray) -> dict[str, float]:
    channel_residuals = [rgb[:, :, idx] - _box_blur(rgb[:, :, idx], radius=1) for idx in range(3)]
    abs_means = np.asarray([np.abs(values).mean() for values in channel_residuals], dtype=np.float32)
    chroma_residual = 0.5 * (
        np.abs(channel_residuals[0] - channel_residuals[1])
        + np.abs(channel_residuals[2] - channel_residuals[1])
    )
    luma_abs_mean = float(np.abs(gray_residual).mean())
    return {
        "residual_rg_corr": _safe_corr(channel_residuals[0], channel_residuals[1]),
        "residual_gb_corr": _safe_corr(channel_residuals[1], channel_residuals[2]),
        "residual_rb_corr": _safe_corr(channel_residuals[0], channel_residuals[2]),
        "residual_rgb_abs_std_ratio": float(
            abs_means.std() / np.clip(float(abs_means.mean()), 1e-8, None)
        ),
        "residual_chroma_luma_abs_ratio": float(
            chroma_residual.mean() / np.clip(luma_abs_mean, 1e-8, None)
        ),
    }


def _noise_v3_extra_features(
    rgb: np.ndarray,
    residual: np.ndarray,
    ela_abs: np.ndarray,
) -> dict[str, float]:
    jpeg_q70_abs = np.mean(_jpeg_reencode_diff(rgb, quality=70), axis=2)
    jpeg_q95_abs = np.mean(_jpeg_reencode_diff(rgb, quality=95), axis=2)
    q70_mean = float(jpeg_q70_abs.mean())
    q95_mean = float(jpeg_q95_abs.mean())
    ela_mean = float(ela_abs.mean())
    values = {
        "jpeg_q70_abs_mean": q70_mean,
        "jpeg_q70_abs_p95": float(np.percentile(jpeg_q70_abs, 95)),
        "jpeg_q95_abs_mean": q95_mean,
        "jpeg_q95_abs_p95": float(np.percentile(jpeg_q95_abs, 95)),
        "jpeg_q70_q95_mean_ratio": float(q70_mean / np.clip(q95_mean, 1e-8, None)),
        "jpeg_q90_q95_mean_delta": float(ela_mean - q95_mean),
    }
    values.update(_phase8_stats(np.abs(residual), "residual"))
    values.update(_phase8_stats(ela_abs, "ela"))
    values.update(_rgb_residual_features(rgb, residual))
    values.update(_tile_std_stats(np.abs(residual), tile_size=16, prefix="residual_tile16"))
    return values


def _noise_feature_values(
    rgb: np.ndarray,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
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
    return values, gray, residual, ela_abs


def extract_noise_features(path: str | Path, image_size: int = 128) -> np.ndarray:
    rgb = _load_rgb(path, image_size)
    values, _gray_values, _residual, _ela_abs = _noise_feature_values(rgb)
    return np.asarray([values[name] for name in NOISE_FEATURE_NAMES], dtype=np.float32)


def extract_noise_v2_features(path: str | Path, image_size: int = 128) -> np.ndarray:
    rgb = _load_rgb(path, image_size)
    values, gray, residual, _ela_abs = _noise_feature_values(rgb)
    extra = _noise_v2_extra_features(gray, residual)
    values.update(extra)
    return np.asarray([values[name] for name in NOISE_V2_FEATURE_NAMES], dtype=np.float32)


def extract_noise_v3_features(path: str | Path, image_size: int = 128) -> np.ndarray:
    rgb = _load_rgb(path, image_size)
    values, gray, residual, ela_abs = _noise_feature_values(rgb)
    values.update(_noise_v2_extra_features(gray, residual))
    values.update(_noise_v3_extra_features(rgb, residual, ela_abs))
    return np.asarray([values[name] for name in NOISE_V3_FEATURE_NAMES], dtype=np.float32)


def feature_names(feature_set: str) -> list[str]:
    if feature_set == "photometric":
        return list(PHOTOMETRIC_FEATURE_NAMES)
    if feature_set == "noise":
        return list(NOISE_FEATURE_NAMES)
    if feature_set == "noise_v2":
        return list(NOISE_V2_FEATURE_NAMES)
    if feature_set == "noise_v3":
        return list(NOISE_V3_FEATURE_NAMES)
    if feature_set == "combined":
        return list(PHOTOMETRIC_FEATURE_NAMES) + list(NOISE_FEATURE_NAMES)
    if feature_set == "combined_v2":
        return list(PHOTOMETRIC_FEATURE_NAMES) + list(NOISE_V2_FEATURE_NAMES)
    if feature_set == "combined_v3":
        return list(PHOTOMETRIC_FEATURE_NAMES) + list(NOISE_V3_FEATURE_NAMES)
    raise ValueError(f"Unsupported feature set: {feature_set}")


def extract_feature_set(path: str | Path, image_size: int, feature_set: str) -> np.ndarray:
    if feature_set == "photometric":
        return extract_photometric_features(path, image_size=image_size)
    if feature_set == "noise":
        return extract_noise_features(path, image_size=image_size)
    if feature_set == "noise_v2":
        return extract_noise_v2_features(path, image_size=image_size)
    if feature_set == "noise_v3":
        return extract_noise_v3_features(path, image_size=image_size)
    if feature_set == "combined":
        return np.concatenate(
            [
                extract_photometric_features(path, image_size=image_size),
                extract_noise_features(path, image_size=image_size),
            ]
        )
    if feature_set == "combined_v2":
        return np.concatenate(
            [
                extract_photometric_features(path, image_size=image_size),
                extract_noise_v2_features(path, image_size=image_size),
            ]
        )
    if feature_set == "combined_v3":
        return np.concatenate(
            [
                extract_photometric_features(path, image_size=image_size),
                extract_noise_v3_features(path, image_size=image_size),
            ]
        )
    raise ValueError(f"Unsupported feature set: {feature_set}")
