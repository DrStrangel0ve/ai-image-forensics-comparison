from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forensic_compare.datasets import stable_path_score
from forensic_compare.foundation import (
    build_frozen_encoder,
    encode_batch,
    frozen_encoder_transform,
    supported_frozen_encoders,
)
from forensic_compare.freuid import apcer_at_bpcer, freuid_competition_path, freuid_metrics
from forensic_compare.metrics import binary_metrics
from forensic_compare.utils import ensure_dir, resolve_device, seed_everything, write_json


ENCODERS = list(supported_frozen_encoders())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a frozen image-encoder baseline on FREUID CSV metadata.")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--image-root", default="data/raw/freuid_2026/images")
    parser.add_argument("--output-dir", default="runs/freuid_frozen_encoder")
    parser.add_argument("--encoder", choices=ENCODERS, default="convnext_tiny")
    parser.add_argument("--pretrained", action="store_true", help="Use public pretrained encoder weights.")
    parser.add_argument(
        "--classifier",
        choices=["logistic_regression", "mlp"],
        default="logistic_regression",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument(
        "--embedding-cache-dir",
        default=None,
        help="Optional directory for cached per-image frozen encoder embeddings.",
    )
    parser.add_argument("--skip-errors", action="store_true")
    return parser.parse_args()


def _resolve_image_path(image_root: Path, image_path: object) -> Path:
    raw = str(image_path)
    candidates = [
        image_root / freuid_competition_path(raw),
        image_root / raw.replace("\\", "/"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find FREUID image for {raw}; tried {candidates}")


def _limit_frame(frame: pd.DataFrame, max_samples: int, seed: int) -> pd.DataFrame:
    if max_samples <= 0 or max_samples >= len(frame):
        return frame.reset_index(drop=True)
    rows = []
    per_label = max(1, max_samples // max(1, frame["label"].nunique()))
    for label, group in frame.groupby("label", sort=True):
        ordered = group.assign(_score=group["id"].astype(str).map(lambda value: stable_path_score(value, seed)))
        rows.append(ordered.sort_values(["_score", "id"]).head(per_label).drop(columns=["_score"]))
    limited = pd.concat(rows, ignore_index=True)
    if len(limited) < max_samples:
        remaining = frame[~frame["id"].isin(set(limited["id"]))].copy()
        remaining["_score"] = remaining["id"].astype(str).map(lambda value: stable_path_score(value, seed + 17))
        limited = pd.concat(
            [limited, remaining.sort_values(["_score", "id"]).head(max_samples - len(limited)).drop(columns=["_score"])],
            ignore_index=True,
        )
    return limited.head(max_samples).reset_index(drop=True)


def _prepare_rows(
    frame: pd.DataFrame,
    image_root: Path,
    max_samples: int,
    seed: int,
    skip_errors: bool,
) -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    frame = _limit_frame(frame, max_samples, seed)
    rows: list[dict[str, object]] = []
    skipped: list[dict[str, str]] = []
    for row in frame.to_dict("records"):
        try:
            local_path = _resolve_image_path(image_root, row["image_path"])
            if skip_errors:
                with Image.open(local_path) as image:
                    image.verify()
        except Exception as exc:
            if not skip_errors:
                raise
            skipped.append({"id": str(row.get("id", "")), "image_path": str(row.get("image_path", "")), "error": repr(exc)})
            continue
        rows.append(
            {
                "id": str(row["id"]),
                "image_path": str(row["image_path"]),
                "local_path": str(local_path),
                "y_true": int(row["label"]),
                "type": str(row.get("type", "")),
            }
        )
    if not rows:
        raise ValueError("No usable FREUID rows after resolving image paths")
    return rows, skipped


def _embedding_cache_path(
    cache_dir: Path,
    local_path: Path,
    encoder_name: str,
    weights: str | None,
    image_size: int,
) -> Path:
    stat = local_path.stat()
    key = "|".join(
        [
            str(local_path.resolve()),
            str(stat.st_size),
            str(stat.st_mtime_ns),
            encoder_name,
            str(weights),
            str(image_size),
        ]
    )
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return cache_dir / encoder_name / str(image_size) / f"{digest}.npy"


def _load_image_tensor(path: Path, transform) -> torch.Tensor:
    with Image.open(path) as image:
        return transform(image.convert("RGB"))


@torch.no_grad()
def _embedding_matrix(
    rows: list[dict[str, object]],
    encoder,
    transform,
    encoder_name: str,
    weights: str | None,
    image_size: int,
    batch_size: int,
    device: torch.device,
    cache_dir: Path | None,
    desc: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    batch_size = max(1, int(batch_size))
    encoder = encoder.to(device)
    features: list[np.ndarray | None] = [None] * len(rows)
    cache_stats = {"hits": 0, "misses": 0}
    pending_images: list[torch.Tensor] = []
    pending_positions: list[int] = []
    pending_cache_paths: list[Path | None] = []

    def flush() -> int:
        if not pending_images:
            return 0
        n_flushed = len(pending_images)
        batch = torch.stack(pending_images).to(device)
        embeddings = encode_batch(encoder, batch).detach().cpu().numpy()
        for position, embedding, cache_path in zip(pending_positions, embeddings, pending_cache_paths):
            embedding = np.asarray(embedding, dtype=np.float32)
            features[position] = embedding
            if cache_path is not None:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                np.save(cache_path, embedding)
        pending_images.clear()
        pending_positions.clear()
        pending_cache_paths.clear()
        return n_flushed

    progress = tqdm(total=len(rows), desc=desc)
    try:
        for index, row in enumerate(rows):
            local_path = Path(str(row["local_path"]))
            cache_path = (
                _embedding_cache_path(cache_dir, local_path, encoder_name, weights, image_size)
                if cache_dir is not None
                else None
            )
            if cache_path is not None and cache_path.exists():
                features[index] = np.load(cache_path)
                cache_stats["hits"] += 1
                progress.update(1)
                continue
            pending_images.append(_load_image_tensor(local_path, transform))
            pending_positions.append(index)
            pending_cache_paths.append(cache_path)
            cache_stats["misses"] += 1
            if len(pending_images) >= batch_size:
                progress.update(flush())
    finally:
        progress.update(flush())
        progress.close()

    missing = [index for index, feature in enumerate(features) if feature is None]
    if missing:
        raise RuntimeError(f"Failed to encode {len(missing)} rows; first missing index is {missing[0]}")
    return (
        np.vstack([np.asarray(feature, dtype=np.float32) for feature in features if feature is not None]),
        np.asarray([int(row["y_true"]) for row in rows], dtype=int),
        cache_stats,
    )


def _classifier(name: str, seed: int):
    if name == "logistic_regression":
        return Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("model", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)),
            ]
        )
    if name == "mlp":
        return Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(128,),
                        activation="relu",
                        alpha=1e-4,
                        max_iter=600,
                        random_state=seed,
                        early_stopping=True,
                    ),
                ),
            ]
        )
    raise ValueError(f"Unsupported classifier: {name}")


def _scores(model, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(features)[:, 1], dtype=float)
    raw = np.asarray(model.decision_function(features), dtype=float)
    return 1.0 / (1.0 + np.exp(-raw))


def run_baseline(args: argparse.Namespace) -> dict[str, object]:
    seed_everything(args.seed)
    output_dir = ensure_dir(args.output_dir)
    image_root = Path(args.image_root)
    device = resolve_device(args.device)
    spec = build_frozen_encoder(args.encoder, pretrained=args.pretrained)
    transform = frozen_encoder_transform(spec.image_size, spec.mean, spec.std)
    cache_dir = Path(args.embedding_cache_dir) if args.embedding_cache_dir else None

    train_rows, train_skipped = _prepare_rows(
        pd.read_csv(args.train_csv), image_root, args.max_train_samples, args.seed, args.skip_errors
    )
    val_rows, val_skipped = _prepare_rows(
        pd.read_csv(args.val_csv), image_root, args.max_val_samples, args.seed + 1, args.skip_errors
    )
    x_train, y_train, train_cache = _embedding_matrix(
        train_rows,
        spec.model,
        transform,
        args.encoder,
        spec.weights,
        spec.image_size,
        args.batch_size,
        device,
        cache_dir,
        "freuid/embeddings/train",
    )
    x_val, y_val, val_cache = _embedding_matrix(
        val_rows,
        spec.model,
        transform,
        args.encoder,
        spec.weights,
        spec.image_size,
        args.batch_size,
        device,
        cache_dir,
        "freuid/embeddings/val",
    )

    model = _classifier(args.classifier, args.seed)
    model.fit(x_train, y_train)
    scores = _scores(model, x_val)
    operating_point = apcer_at_bpcer(y_val, scores, bpcer_target=0.01)
    labels = (scores >= operating_point.threshold).astype(int)

    metrics = binary_metrics(y_val, scores)
    metrics.update(freuid_metrics(y_val, scores))
    metrics.update(
        {
            "method": f"freuid_frozen_encoder_{args.encoder}_{args.classifier}",
            "encoder": args.encoder,
            "pretrained": bool(args.pretrained),
            "weights": spec.weights,
            "classifier": args.classifier,
            "embedding_dim": int(spec.embedding_dim),
            "image_size": int(spec.image_size),
            "device": str(device),
            "n_train": int(len(y_train)),
            "n_val": int(len(y_val)),
            "n_train_skipped": int(len(train_skipped)),
            "n_val_skipped": int(len(val_skipped)),
            "embedding_cache_dir": str(cache_dir) if cache_dir is not None else None,
            "embedding_cache": {
                "train_hits": int(train_cache["hits"]),
                "train_misses": int(train_cache["misses"]),
                "val_hits": int(val_cache["hits"]),
                "val_misses": int(val_cache["misses"]),
            },
            "threshold_for_1pct_bpcer": float(operating_point.threshold),
        }
    )
    write_json(metrics, output_dir / "metrics.json")
    write_json({"train": train_skipped, "val": val_skipped}, output_dir / "skipped.json")
    write_json(
        {
            "encoder": args.encoder,
            "pretrained": bool(args.pretrained),
            "weights": spec.weights,
            "embedding_dim": int(spec.embedding_dim),
            "image_size": int(spec.image_size),
            "mean": list(spec.mean),
            "std": list(spec.std),
        },
        output_dir / "encoder.json",
    )
    joblib.dump(model, output_dir / "classifier.joblib")
    np.savez_compressed(
        output_dir / "embeddings.npz",
        x_train=x_train.astype(np.float32),
        y_train=y_train.astype(np.int64),
        x_val=x_val.astype(np.float32),
        y_val=y_val.astype(np.int64),
    )
    with (output_dir / "val_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["id", "image_path", "local_path", "type", "y_true", "fraud_score", "label"],
        )
        writer.writeheader()
        for row, score, label in zip(val_rows, scores, labels):
            writer.writerow({**row, "fraud_score": float(score), "label": int(label)})
    return metrics


def main() -> None:
    args = parse_args()
    metrics = run_baseline(args)
    print(Path(args.output_dir).resolve())
    print(f"apcer_at_1pct_bpcer={metrics['apcer_at_1pct_bpcer']:.6f}")
    print(f"audet_proxy={metrics['audet_proxy']:.6f}")


if __name__ == "__main__":
    main()
