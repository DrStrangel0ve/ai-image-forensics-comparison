from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forensic_compare.datasets import stable_path_score  # noqa: E402
from forensic_compare.freuid import freuid_competition_path, freuid_metrics  # noqa: E402
from forensic_compare.freuid_model import (  # noqa: E402
    build_freuid_model,
    required_freuid_input_size,
    supported_freuid_models,
)
from forensic_compare.freuid_transforms import DocumentViewTransform  # noqa: E402
from forensic_compare.metrics import binary_metrics  # noqa: E402
from forensic_compare.utils import ensure_dir, resolve_device, seed_everything, write_json  # noqa: E402


RESEARCH_TRACK = "post_freeze_highres_2026_07_13"
AGGREGATION_MODES = ("mean", "mean_max", "mean_max_std")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache high-resolution frozen FREUID embeddings and fit fixed linear probes."
    )
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", choices=supported_freuid_models(), default="dinov2_base_518")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--view-mode", choices=("single", "grid", "five_crop"), default="five_crop")
    parser.add_argument("--grid-rows", type=int, default=0)
    parser.add_argument("--grid-cols", type=int, default=0)
    parser.add_argument("--five-crop-zoom", type=float, default=1.15)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--view-batch-size", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--data-parallel", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-train-samples", type=int, default=12000)
    parser.add_argument("--max-val-samples", type=int, default=4000)
    parser.add_argument("--aggregation", choices=AGGREGATION_MODES, nargs="+", default=list(AGGREGATION_MODES))
    parser.add_argument("--primary-aggregation", choices=AGGREGATION_MODES, default="mean_max_std")
    parser.add_argument("--logistic-c", type=float, default=0.1)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--seed", type=int, default=43)
    return parser.parse_args()


def _resolve_image_path(image_root: Path, raw_path: object) -> Path:
    raw = str(raw_path).replace("\\", "/")
    candidates = [
        Path(raw),
        image_root / raw,
        image_root / freuid_competition_path(raw),
        image_root / Path(raw).name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve FREUID image {raw!r}; tried {candidates}")


def _validated_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"id", "image_path", "label", "type"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    frame = frame.copy()
    frame["id"] = frame["id"].astype(str)
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    frame["type"] = frame["type"].astype(str)
    if frame["id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate ids")
    if set(frame["label"].unique()) != {0, 1}:
        raise ValueError(f"{path} must contain both binary labels")
    return frame.reset_index(drop=True)


def limit_frame(frame: pd.DataFrame, max_samples: int, seed: int) -> pd.DataFrame:
    """Deterministically cap every type/label group before filling spare slots."""

    if max_samples <= 0 or max_samples >= len(frame):
        return frame.reset_index(drop=True)
    scored = frame.copy()
    scored["_score"] = scored["id"].map(lambda value: stable_path_score(value, seed))
    groups = [
        group.sort_values(["_score", "id"], kind="mergesort")
        for _key, group in scored.groupby(["type", "label"], sort=True)
    ]
    selected: list[int] = []
    cursor = 0
    while len(selected) < max_samples:
        progressed = False
        for group in groups:
            if cursor < len(group):
                selected.append(int(group.index[cursor]))
                progressed = True
                if len(selected) >= max_samples:
                    break
        if not progressed:
            break
        cursor += 1
    return scored.loc[selected].drop(columns=["_score"]).reset_index(drop=True)


class FreuidEmbeddingDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        image_root: Path,
        transform: DocumentViewTransform,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.image_root = image_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        path = _resolve_image_path(self.image_root, row["image_path"])
        with Image.open(path) as image:
            views = self.transform(image.convert("RGB"))
        if views.ndim == 3:
            views = views.unsqueeze(0)
        return views, int(row["label"]), str(row["id"]), str(row["type"])


def aggregate_view_embeddings(view_embeddings: np.ndarray, mode: str) -> np.ndarray:
    values = np.asarray(view_embeddings, dtype=np.float32)
    if values.ndim != 3:
        raise ValueError(f"Expected NVD embeddings, got shape {values.shape}")
    mean = values.mean(axis=1)
    if mode == "mean":
        return mean
    maximum = values.max(axis=1)
    if mode == "mean_max":
        return np.concatenate([mean, maximum], axis=1)
    if mode == "mean_max_std":
        return np.concatenate([mean, maximum, values.std(axis=1)], axis=1)
    raise ValueError(f"Unsupported aggregation mode: {mode}")


@torch.inference_mode()
def extract_embeddings(
    encoder: nn.Module,
    loader: DataLoader,
    device: torch.device,
    view_batch_size: int,
    description: str,
) -> dict[str, object]:
    ids: list[str] = []
    labels: list[int] = []
    types: list[str] = []
    batches: list[np.ndarray] = []
    encoder.eval()
    for views, batch_labels, batch_ids, batch_types in tqdm(loader, desc=description):
        batch_size, n_views, channels, height, width = views.shape
        flat = views.reshape(batch_size * n_views, channels, height, width)
        encoded_chunks: list[torch.Tensor] = []
        for start in range(0, len(flat), view_batch_size):
            images = flat[start : start + view_batch_size].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                embeddings = encoder(images)
            if embeddings.ndim > 2:
                embeddings = torch.flatten(embeddings, 1)
            encoded_chunks.append(F.normalize(embeddings.float(), dim=1).cpu())
        batch_embeddings = torch.cat(encoded_chunks, dim=0).reshape(batch_size, n_views, -1)
        batches.append(batch_embeddings.numpy().astype(np.float16))
        ids.extend(str(value) for value in batch_ids)
        labels.extend(int(value) for value in batch_labels.tolist())
        types.extend(str(value) for value in batch_types)
    if not batches:
        raise ValueError("No embeddings were extracted")
    return {
        "ids": np.asarray(ids),
        "labels": np.asarray(labels, dtype=np.int8),
        "types": np.asarray(types),
        "embeddings": np.concatenate(batches, axis=0),
    }


def _write_cache(path: Path, data: dict[str, object]) -> None:
    np.savez(
        path,
        ids=data["ids"],
        labels=data["labels"],
        types=data["types"],
        embeddings=data["embeddings"],
    )


def _metric_row(y_true: np.ndarray, scores: np.ndarray, aggregation: str) -> dict[str, object]:
    metrics = binary_metrics(y_true, scores)
    metrics.update(freuid_metrics(y_true, scores))
    metrics["aggregation"] = aggregation
    metrics["selection_objective"] = float(metrics["audet_proxy"] + 0.25 * metrics["apcer_at_1pct_bpcer"])
    return metrics


def _write_predictions(path: Path, ids: np.ndarray, y_true: np.ndarray, scores: np.ndarray, threshold: float) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "y_true", "fraud_score", "label"])
        writer.writeheader()
        for image_id, target, score in zip(ids, y_true, scores):
            writer.writerow(
                {
                    "id": str(image_id),
                    "y_true": int(target),
                    "fraud_score": f"{float(score):.10f}",
                    "label": int(float(score) >= threshold),
                }
            )


def run(args: argparse.Namespace) -> dict[str, object]:
    seed_everything(args.seed)
    if args.batch_size <= 0 or args.view_batch_size <= 0:
        raise ValueError("Batch sizes must be positive")
    if args.logistic_c <= 0:
        raise ValueError("--logistic-c must be positive")
    if args.primary_aggregation not in args.aggregation:
        raise ValueError("--primary-aggregation must be included in --aggregation")
    required_size = required_freuid_input_size(args.model)
    if required_size is not None and args.image_size != required_size:
        raise ValueError(f"{args.model} requires --image-size {required_size}")

    output_dir = ensure_dir(args.output_dir)
    device = resolve_device(args.device)
    train_frame = limit_frame(_validated_frame(Path(args.train_csv)), args.max_train_samples, args.seed)
    val_frame = limit_frame(_validated_frame(Path(args.val_csv)), args.max_val_samples, args.seed + 1)
    transform = DocumentViewTransform(
        args.image_size,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        view_mode=args.view_mode,
        five_crop_zoom=args.five_crop_zoom,
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_loader = DataLoader(
        FreuidEmbeddingDataset(train_frame, Path(args.image_root), transform),
        **loader_kwargs,
    )
    val_loader = DataLoader(
        FreuidEmbeddingDataset(val_frame, Path(args.image_root), transform),
        **loader_kwargs,
    )

    model = build_freuid_model(
        args.model,
        num_types=max(1, train_frame["type"].nunique()),
        pretrained=args.pretrained,
        freeze_encoder=True,
    )
    encoder: nn.Module = model.encoder.to(device).eval()
    gpu_count = torch.cuda.device_count() if device.type == "cuda" else 0
    if args.data_parallel and gpu_count > 1:
        encoder = nn.DataParallel(encoder)
    train_data = extract_embeddings(
        encoder,
        train_loader,
        device,
        view_batch_size=args.view_batch_size,
        description="freuid/embed/train",
    )
    _write_cache(output_dir / "train_embeddings.npz", train_data)
    val_data = extract_embeddings(
        encoder,
        val_loader,
        device,
        view_batch_size=args.view_batch_size,
        description="freuid/embed/val",
    )
    _write_cache(output_dir / "val_embeddings.npz", val_data)
    del encoder
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    y_train = np.asarray(train_data["labels"], dtype=int)
    y_val = np.asarray(val_data["labels"], dtype=int)
    rows: list[dict[str, object]] = []
    for aggregation in args.aggregation:
        train_features = aggregate_view_embeddings(train_data["embeddings"], aggregation)
        val_features = aggregate_view_embeddings(val_data["embeddings"], aggregation)
        classifier = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        C=args.logistic_c,
                        class_weight="balanced",
                        max_iter=args.max_iterations,
                        random_state=args.seed,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
        classifier.fit(train_features, y_train)
        scores = classifier.predict_proba(val_features)[:, 1]
        metrics = _metric_row(y_val, scores, aggregation)
        rows.append(metrics)
        joblib.dump(classifier, output_dir / f"probe_{aggregation}.joblib")
        _write_predictions(
            output_dir / f"val_predictions_{aggregation}.csv",
            np.asarray(val_data["ids"]),
            y_val,
            scores,
            float(metrics["threshold_at_1pct_bpcer"]),
        )

    primary = next(row for row in rows if row["aggregation"] == args.primary_aggregation)
    summary = {
        "research_track": RESEARCH_TRACK,
        "competition_eligibility": "post_freeze_research_only",
        "model": args.model,
        "pretrained": bool(args.pretrained),
        "image_size": int(args.image_size),
        "view_mode": args.view_mode,
        "five_crop_zoom": float(args.five_crop_zoom),
        "n_views": int(np.asarray(train_data["embeddings"]).shape[1]),
        "embedding_dim": int(np.asarray(train_data["embeddings"]).shape[2]),
        "aggregation_modes": list(args.aggregation),
        "primary_aggregation": args.primary_aggregation,
        "logistic_c": float(args.logistic_c),
        "seed": int(args.seed),
        "output_dir": str(output_dir),
        "device": str(device),
        "gpu_count": int(gpu_count),
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "train_label_counts": {str(key): int(value) for key, value in Counter(y_train).items()},
        "val_label_counts": {str(key): int(value) for key, value in Counter(y_val).items()},
        "train_type_counts": {str(key): int(value) for key, value in Counter(train_data["types"]).items()},
        "val_type_counts": {str(key): int(value) for key, value in Counter(val_data["types"]).items()},
        "rows": rows,
        "primary": primary,
    }
    write_json(summary, output_dir / "summary.json")
    return summary


def main() -> None:
    summary = run(parse_args())
    primary = summary["primary"]
    print(Path(str(summary.get("output_dir", ""))).resolve() if summary.get("output_dir") else "")
    print(
        f"primary={summary['primary_aggregation']} "
        f"audet={primary['audet_proxy']:.6f} "
        f"apcer={primary['apcer_at_1pct_bpcer']:.6f} "
        f"auc={primary['roc_auc']:.6f}"
    )


if __name__ == "__main__":
    main()
