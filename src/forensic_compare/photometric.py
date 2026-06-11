from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

FEATURE_NAMES = [
    "gray_mean",
    "gray_std",
    "gray_skew",
    "edge_mean",
    "edge_std",
    "edge_p90",
    "edge_p99",
    "normal_z_mean",
    "normal_z_std",
    "normal_z_p05",
    "normal_xy_std",
    "slope_mean",
    "slope_std",
    "integrability_abs_mean",
    "integrability_std",
    "integrability_p95",
    "laplacian_abs_mean",
    "laplacian_std",
    "high_frequency_std",
    "saturation_mean",
    "saturation_std",
    "edge_saturation_corr",
    "rg_corr",
    "rb_corr",
    "gb_corr",
    "normal_z_entropy",
]


def _load_rgb(path: str | Path, image_size: int) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32) / 255.0


def _safe_skew(values: np.ndarray) -> float:
    flat = values.reshape(-1)
    std = float(flat.std())
    if std < 1e-8:
        return 0.0
    return float(np.mean(((flat - flat.mean()) / std) ** 3))


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    if float(a_flat.std()) < 1e-8 or float(b_flat.std()) < 1e-8:
        return 0.0
    return float(np.corrcoef(a_flat, b_flat)[0, 1])


def _box_blur_3x3(values: np.ndarray) -> np.ndarray:
    padded = np.pad(values, 1, mode="reflect")
    acc = np.zeros_like(values)
    for dy in range(3):
        for dx in range(3):
            acc += padded[dy : dy + values.shape[0], dx : dx + values.shape[1]]
    return acc / 9.0


def estimate_pseudo_normals(gray: np.ndarray, strength: float = 8.0) -> np.ndarray:
    """Estimate a single-image normal field from brightness gradients.

    This is a shape-from-shading proxy, not calibrated multi-light photometric
    stereo. It still exposes useful normal-consistency artifacts for a baseline.
    """

    gy, gx = np.gradient(gray.astype(np.float32))
    nx = -gx * strength
    ny = -gy * strength
    nz = np.ones_like(gray, dtype=np.float32)
    normals = np.stack([nx, ny, nz], axis=-1)
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    return normals / np.clip(norms, 1e-8, None)


def extract_features(path: str | Path, image_size: int = 128) -> np.ndarray:
    rgb = _load_rgb(path, image_size=image_size)
    gray = (
        0.2126 * rgb[:, :, 0]
        + 0.7152 * rgb[:, :, 1]
        + 0.0722 * rgb[:, :, 2]
    ).astype(np.float32)

    gy, gx = np.gradient(gray)
    edge = np.sqrt(gx**2 + gy**2)
    normals = estimate_pseudo_normals(gray)
    nx = normals[:, :, 0]
    ny = normals[:, :, 1]
    nz = normals[:, :, 2]

    p = -nx / np.clip(nz, 1e-6, None)
    q = -ny / np.clip(nz, 1e-6, None)
    dq_dx = np.gradient(q, axis=1)
    dp_dy = np.gradient(p, axis=0)
    integrability = dp_dy - dq_dx

    laplacian = (
        -4.0 * gray
        + np.roll(gray, 1, axis=0)
        + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1)
        + np.roll(gray, -1, axis=1)
    )
    high_frequency = gray - _box_blur_3x3(gray)

    max_rgb = rgb.max(axis=2)
    min_rgb = rgb.min(axis=2)
    saturation = (max_rgb - min_rgb) / np.clip(max_rgb, 1e-6, None)

    nz_hist, _ = np.histogram(nz, bins=16, range=(0.0, 1.0), density=False)
    nz_probs = nz_hist.astype(np.float64) / max(1, nz_hist.sum())
    nz_entropy = -float(np.sum(nz_probs * np.log2(np.clip(nz_probs, 1e-12, None))))

    values = {
        "gray_mean": float(gray.mean()),
        "gray_std": float(gray.std()),
        "gray_skew": _safe_skew(gray),
        "edge_mean": float(edge.mean()),
        "edge_std": float(edge.std()),
        "edge_p90": float(np.percentile(edge, 90)),
        "edge_p99": float(np.percentile(edge, 99)),
        "normal_z_mean": float(nz.mean()),
        "normal_z_std": float(nz.std()),
        "normal_z_p05": float(np.percentile(nz, 5)),
        "normal_xy_std": float(np.sqrt(nx**2 + ny**2).std()),
        "slope_mean": float(np.sqrt(p**2 + q**2).mean()),
        "slope_std": float(np.sqrt(p**2 + q**2).std()),
        "integrability_abs_mean": float(np.abs(integrability).mean()),
        "integrability_std": float(integrability.std()),
        "integrability_p95": float(np.percentile(np.abs(integrability), 95)),
        "laplacian_abs_mean": float(np.abs(laplacian).mean()),
        "laplacian_std": float(laplacian.std()),
        "high_frequency_std": float(high_frequency.std()),
        "saturation_mean": float(saturation.mean()),
        "saturation_std": float(saturation.std()),
        "edge_saturation_corr": _safe_corr(edge, saturation),
        "rg_corr": _safe_corr(rgb[:, :, 0], rgb[:, :, 1]),
        "rb_corr": _safe_corr(rgb[:, :, 0], rgb[:, :, 2]),
        "gb_corr": _safe_corr(rgb[:, :, 1], rgb[:, :, 2]),
        "normal_z_entropy": nz_entropy,
    }
    return np.asarray([values[name] for name in FEATURE_NAMES], dtype=np.float32)
