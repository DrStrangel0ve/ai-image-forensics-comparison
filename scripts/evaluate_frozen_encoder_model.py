from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import datasets
from tqdm import tqdm

from forensic_compare.catalog import load_catalog
from forensic_compare.datasets import class_kind, discover_layout, stable_path_score
from forensic_compare.foundation import build_frozen_encoder, encode_batch, frozen_encoder_transform
from forensic_compare.metrics import binary_metrics
from forensic_compare.utils import ensure_dir, read_json, resolve_device, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved frozen-encoder baseline.")
    parser.add_argument("--model-dir", required=True, help="Directory containing classifier.joblib.")
    parser.add_argument("--target-key", default=None, help="Dataset key from configs/datasets.json.")
    parser.add_argument("--target-dir", default=None, help="Target dataset folder. Overrides --target-key.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--target-split", choices=["all", "test"], default="all")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    return parser.parse_args()


def _target_dir(args: argparse.Namespace) -> Path:
    if args.target_dir:
        return Path(args.target_dir)
    if not args.target_key:
        raise SystemExit("Provide --target-key or --target-dir")
    return load_catalog()[args.target_key].local_dir


def _fake_class_index(class_to_idx: dict[str, int]) -> int:
    for class_name, idx in class_to_idx.items():
        if class_kind(class_name) == "fake":
            return idx
    raise ValueError(f"Could not identify fake/generated class in {class_to_idx}")


def _stable_test_indices(dataset: datasets.ImageFolder, test_fraction: float, seed: int) -> list[int]:
    targets = [label for _path, label in dataset.samples]
    indices: list[int] = []
    for label in sorted(set(targets)):
        label_indices = [idx for idx, target in enumerate(targets) if target == label]
        label_indices = sorted(
            label_indices,
            key=lambda idx: stable_path_score(dataset.samples[idx][0], seed),
        )
        n_test = max(1, int(round(len(label_indices) * test_fraction)))
        indices.extend(label_indices[:n_test])
    return indices


def _dataset(target_dir: Path, transform, args: argparse.Namespace):
    layout = discover_layout(target_dir)
    if layout.single:
        dataset = datasets.ImageFolder(layout.single, transform=transform)
        if args.target_split == "test":
            dataset = Subset(dataset, _stable_test_indices(dataset, args.val_fraction, args.seed))
        return dataset, layout
    if layout.train and layout.test:
        test_dataset = datasets.ImageFolder(layout.test, transform=transform)
        if args.target_split == "test":
            return test_dataset, layout
        train_dataset = datasets.ImageFolder(layout.train, transform=transform)
        return ConcatDataset([train_dataset, test_dataset]), layout
    raise ValueError(f"Unsupported dataset layout: {layout}")


def _class_to_idx(dataset) -> dict[str, int]:
    if isinstance(dataset, Subset):
        return dataset.dataset.class_to_idx
    if isinstance(dataset, ConcatDataset):
        return dataset.datasets[0].class_to_idx
    return dataset.class_to_idx


def _dataset_path(dataset, index: int) -> str:
    if isinstance(dataset, Subset):
        return _dataset_path(dataset.dataset, dataset.indices[index])
    if isinstance(dataset, ConcatDataset):
        offset = index
        for child in dataset.datasets:
            if offset < len(child):
                return _dataset_path(child, offset)
            offset -= len(child)
        raise IndexError(index)
    return str(dataset.samples[index][0])


@torch.no_grad()
def _embedding_matrix(dataset, encoder, target_fake_idx: int, args: argparse.Namespace):
    device = resolve_device(args.device)
    encoder = encoder.to(device)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    paths: list[str] = []
    dataset_offset = 0
    for images, batch_labels in tqdm(loader, desc="embeddings/target"):
        images = images.to(device)
        embeddings = encode_batch(encoder, images).detach().cpu().numpy()
        labels_np = batch_labels.numpy()
        features.append(embeddings)
        labels.append((labels_np == target_fake_idx).astype(int))
        paths.extend(
            _dataset_path(loader.dataset, index)
            for index in range(dataset_offset, dataset_offset + len(labels_np))
        )
        dataset_offset += len(labels_np)
    if not features:
        raise ValueError("No target embeddings extracted")
    return np.vstack(features), np.concatenate(labels), paths


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    output_dir = ensure_dir(args.output_dir)
    source_metrics = read_json(model_dir / "metrics.json")
    encoder_name = source_metrics["encoder"]
    pretrained = bool(source_metrics.get("pretrained", False))
    spec = build_frozen_encoder(encoder_name, pretrained=pretrained)
    transform = frozen_encoder_transform(spec.image_size, spec.mean, spec.std)
    target_dataset, layout = _dataset(_target_dir(args), transform, args)
    target_class_to_idx = _class_to_idx(target_dataset)
    target_fake_idx = _fake_class_index(target_class_to_idx)

    x_target, y_target, target_paths = _embedding_matrix(
        target_dataset, spec.model, target_fake_idx, args
    )
    expected_dim = int(source_metrics["embedding_dim"])
    if x_target.shape[1] != expected_dim:
        raise ValueError(
            f"Embedding dimension mismatch: source expected {expected_dim}, got {x_target.shape[1]}"
        )

    classifier = joblib.load(model_dir / "classifier.joblib")
    if hasattr(classifier, "predict_proba"):
        scores = classifier.predict_proba(x_target)[:, 1]
    else:
        raw_scores = classifier.decision_function(x_target)
        scores = 1.0 / (1.0 + np.exp(-raw_scores))

    metrics = binary_metrics(y_target, scores)
    metrics.update(
        {
            "method": "cross_frozen_encoder",
            "source_model_dir": str(model_dir),
            "target_dir": str(_target_dir(args)),
            "target_split": args.target_split,
            "encoder": encoder_name,
            "pretrained": pretrained,
            "weights": spec.weights,
            "classifier": source_metrics["classifier"],
            "embedding_dim": int(spec.embedding_dim),
            "image_size": int(spec.image_size),
            "device": str(resolve_device(args.device)),
            "target_fake_idx": int(target_fake_idx),
            "target_class_to_idx": target_class_to_idx,
            "n_target": len(target_dataset),
            "layout": {
                "train": str(layout.train) if layout.train else None,
                "test": str(layout.test) if layout.test else None,
                "single": str(layout.single) if layout.single else None,
            },
        }
    )
    write_json(metrics, output_dir / "metrics.json")
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "y_true", "fake_score"])
        writer.writeheader()
        for path, truth, score in zip(target_paths, y_target, scores):
            writer.writerow({"path": str(path), "y_true": int(truth), "fake_score": float(score)})
    print(f"Wrote cross-dataset frozen-encoder evaluation to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
