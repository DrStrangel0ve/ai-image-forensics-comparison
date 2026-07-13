from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from forensic_compare.utils import resolve_device, write_json  # noqa: E402
from infer_freuid_finetune import _image_paths, score_checkpoint  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a memory-bounded ensemble of frozen FREUID checkpoints.")
    parser.add_argument("--input-dir", default="/data")
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--weight", action="append", type=float, required=True)
    parser.add_argument("--normalization", choices=["raw", "rank"], default="rank")
    parser.add_argument("--output-csv", default="/submissions/submission.csv")
    parser.add_argument("--manifest-out", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--max-images", type=int, default=0)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rank_percentiles(values: np.ndarray) -> np.ndarray:
    """Match pandas average percentage ranks without requiring pandas at runtime."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("rank_percentiles expects a non-empty one-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("Cannot rank non-finite scores")
    order = np.argsort(values, kind="mergesort")
    ranked = np.empty(values.size, dtype=float)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        average_one_based_rank = 0.5 * ((start + 1) + end)
        ranked[order[start:end]] = average_one_based_rank / values.size
        start = end
    return ranked


def _normalized_weights(weights: list[float], n_checkpoints: int) -> np.ndarray:
    if len(weights) != n_checkpoints:
        raise ValueError("Provide exactly one --weight for each --checkpoint")
    result = np.asarray(weights, dtype=float)
    if not np.isfinite(result).all() or (result < 0).any() or float(result.sum()) <= 0:
        raise ValueError("Ensemble weights must be finite, non-negative, and have a positive sum")
    return result / result.sum()


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, object]:
    checkpoint_paths = [Path(value) for value in args.checkpoint]
    weights = _normalized_weights(list(args.weight), len(checkpoint_paths))
    device = resolve_device(args.device)
    paths = _image_paths(Path(args.input_dir), args.recursive, args.max_images)

    expected_ids: list[str] | None = None
    score_columns: list[np.ndarray] = []
    members: list[dict[str, object]] = []
    for checkpoint_path, weight in zip(checkpoint_paths, weights):
        ids, scores, metadata = score_checkpoint(
            checkpoint_path,
            paths,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        if expected_ids is None:
            expected_ids = ids
        elif ids != expected_ids:
            raise ValueError(f"Checkpoint {checkpoint_path} returned a different id order")
        if args.normalization == "rank":
            scores = rank_percentiles(scores)
        score_columns.append(scores)
        members.append(
            {
                **metadata,
                "bytes": int(checkpoint_path.stat().st_size),
                "sha256": _sha256(checkpoint_path),
                "weight": float(weight),
            }
        )

    assert expected_ids is not None
    score_matrix = np.column_stack(score_columns)
    fused_scores = score_matrix @ weights
    if not np.isfinite(fused_scores).all() or not ((0.0 <= fused_scores) & (fused_scores <= 1.0)).all():
        raise ValueError("Fused scores must be finite and within [0, 1]")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "label"])
        writer.writeheader()
        for image_id, score in zip(expected_ids, fused_scores):
            writer.writerow({"id": image_id, "label": f"{score:.10f}"})

    manifest_out = Path(args.manifest_out) if args.manifest_out else output_csv.with_suffix(".manifest.json")
    manifest = {
        "runtime": "sequential_checkpoint_ensemble",
        "normalization": args.normalization,
        "members": members,
        "input_dir": str(args.input_dir),
        "output_csv": str(output_csv),
        "device": str(device),
        "n_images": len(expected_ids),
        "score_min": float(fused_scores.min()),
        "score_max": float(fused_scores.max()),
        "score_mean": float(fused_scores.mean()),
    }
    write_json(manifest, manifest_out)
    return manifest


def main() -> None:
    manifest = run(parse_args())
    print(Path(str(manifest["output_csv"])).resolve())
    print(f"n_images={manifest['n_images']} score_mean={manifest['score_mean']:.6f}")


if __name__ == "__main__":
    main()
