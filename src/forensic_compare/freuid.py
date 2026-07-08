from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

import numpy as np


@dataclass(frozen=True)
class FreuidOperatingPoint:
    threshold: float
    bpcer: float
    apcer: float
    n_bona_fide: int
    n_attack: int


def freuid_competition_path(value: object, split: str | None = None) -> str:
    """Return the Kaggle archive path for a FREUID image.

    The labels expose paths such as ``train/<id>.jpeg``, while Kaggle stores
    downloadable files under nested split folders, for example
    ``train/train/<id>.jpeg`` and ``public_test/public_test/<id>.jpeg``.
    """

    raw = str(value).replace("\\", "/").strip()
    if not raw:
        raise ValueError("FREUID image path/id cannot be empty")
    parts = PurePosixPath(raw).parts
    name = parts[-1]
    if "." not in name:
        name = f"{name}.jpeg"
    inferred_split = split
    if inferred_split is None and parts:
        if parts[0] in {"train", "public_test"}:
            inferred_split = parts[0]
    inferred_split = inferred_split or "public_test"
    if inferred_split not in {"train", "public_test"}:
        raise ValueError("FREUID split must be 'train' or 'public_test'")
    return f"{inferred_split}/{inferred_split}/{name}"


def _binary_arrays(y_true: np.ndarray | list[int], scores: np.ndarray | list[float]) -> tuple[np.ndarray, np.ndarray]:
    y_true_arr = np.asarray(y_true, dtype=int)
    score_arr = np.asarray(scores, dtype=float)
    if y_true_arr.shape[0] != score_arr.shape[0]:
        raise ValueError(f"y_true and scores must have the same length, got {len(y_true_arr)} and {len(score_arr)}")
    if len(y_true_arr) == 0:
        raise ValueError("FREUID metrics require at least one sample")
    if not np.isin(y_true_arr, [0, 1]).all():
        raise ValueError("FREUID metrics require binary labels encoded as 0/1")
    if not np.isfinite(score_arr).all():
        raise ValueError("FREUID metrics require finite fraud scores")
    return y_true_arr, score_arr


def apcer_at_bpcer(
    y_true: np.ndarray | list[int],
    scores: np.ndarray | list[float],
    bpcer_target: float = 0.01,
) -> FreuidOperatingPoint:
    """Return APCER at a maximum BPCER target.

    Assumptions used for local validation:
    - label 0 is bona fide / genuine;
    - label 1 is attack / fraud;
    - higher score means more likely fraud;
    - predicted fraud iff score >= threshold.
    """

    if not 0.0 <= bpcer_target <= 1.0:
        raise ValueError("bpcer_target must be in [0, 1]")
    y_true_arr, score_arr = _binary_arrays(y_true, scores)
    bona_scores = score_arr[y_true_arr == 0]
    attack_scores = score_arr[y_true_arr == 1]
    if len(bona_scores) == 0 or len(attack_scores) == 0:
        raise ValueError("FREUID APCER/BPCER metrics require both label classes")

    candidates = np.unique(np.concatenate([score_arr, np.nextafter(score_arr, np.inf), [-np.inf, np.inf]]))
    best: FreuidOperatingPoint | None = None
    for threshold in np.sort(candidates):
        bpcer = float(np.mean(bona_scores >= threshold))
        if bpcer > bpcer_target:
            continue
        apcer = float(np.mean(attack_scores < threshold))
        point = FreuidOperatingPoint(
            threshold=float(threshold),
            bpcer=bpcer,
            apcer=apcer,
            n_bona_fide=int(len(bona_scores)),
            n_attack=int(len(attack_scores)),
        )
        if best is None or point.apcer < best.apcer or (point.apcer == best.apcer and point.bpcer > best.bpcer):
            best = point
    if best is None:
        raise RuntimeError("No threshold satisfied the requested BPCER target")
    return best


def det_curve_frame(y_true: np.ndarray | list[int], scores: np.ndarray | list[float]) -> list[dict[str, float]]:
    y_true_arr, score_arr = _binary_arrays(y_true, scores)
    bona_scores = score_arr[y_true_arr == 0]
    attack_scores = score_arr[y_true_arr == 1]
    if len(bona_scores) == 0 or len(attack_scores) == 0:
        raise ValueError("FREUID DET metrics require both label classes")
    candidates = np.unique(np.concatenate([score_arr, np.nextafter(score_arr, np.inf), [-np.inf, np.inf]]))
    rows = []
    for threshold in np.sort(candidates):
        rows.append(
            {
                "threshold": float(threshold),
                "bpcer": float(np.mean(bona_scores >= threshold)),
                "apcer": float(np.mean(attack_scores < threshold)),
            }
        )
    return rows


def audet_proxy(y_true: np.ndarray | list[int], scores: np.ndarray | list[float]) -> float:
    """Approximate DET area for local model selection.

    Kaggle's official AuDET implementation is authoritative. This proxy integrates
    APCER over BPCER from the local threshold sweep and is only used for offline
    ranking while iterating.
    """

    rows = det_curve_frame(y_true, scores)
    points = sorted({(float(row["bpcer"]), float(row["apcer"])) for row in rows})
    bpcer = np.asarray([point[0] for point in points], dtype=float)
    apcer = np.asarray([point[1] for point in points], dtype=float)
    if len(bpcer) < 2:
        return 0.0
    return float(np.sum(np.diff(bpcer) * (apcer[:-1] + apcer[1:]) * 0.5))


def freuid_metrics(y_true: np.ndarray | list[int], scores: np.ndarray | list[float]) -> dict[str, Any]:
    point = apcer_at_bpcer(y_true, scores, bpcer_target=0.01)
    return {
        "apcer_at_1pct_bpcer": point.apcer,
        "bpcer_at_operating_point": point.bpcer,
        "threshold_at_1pct_bpcer": point.threshold,
        "audet_proxy": audet_proxy(y_true, scores),
        "n_bona_fide": point.n_bona_fide,
        "n_attack": point.n_attack,
    }
