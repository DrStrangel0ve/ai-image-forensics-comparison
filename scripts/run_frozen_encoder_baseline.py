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
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from tqdm import tqdm

from forensic_compare.datasets import class_kind, discover_layout, stable_path_score
from forensic_compare.foundation import (
    build_frozen_encoder,
    encode_batch,
    frozen_encoder_transform,
    supported_frozen_encoders,
)
from forensic_compare.metrics import binary_metrics
from forensic_compare.utils import ensure_dir, resolve_device, seed_everything, write_json


ENCODERS = list(supported_frozen_encoders())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a frozen pretrained-image-encoder baseline for real-vs-generated detection."
    )
    parser.add_argument("--data-dir", default="data/raw/cifake")
    parser.add_argument("--output-dir", default="runs/frozen_encoder")
    parser.add_argument("--encoder", choices=ENCODERS, default="convnext_tiny")
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained encoder weights.")
    parser.add_argument(
        "--classifier",
        choices=["logistic_regression", "mlp"],
        default="logistic_regression",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    return parser.parse_args()


def _targets(dataset: datasets.ImageFolder) -> list[int]:
    return [sample[1] for sample in dataset.samples]


def _balanced_subset_indices(
    targets: list[int],
    max_samples: int,
    seed: int,
    paths: list[str] | None = None,
) -> list[int]:
    if max_samples <= 0 or max_samples >= len(targets):
        return list(range(len(targets)))
    labels = sorted(set(targets))
    per_label = max(1, max_samples // max(1, len(labels)))
    selected: list[int] = []
    for label in labels:
        label_indices = [idx for idx, target in enumerate(targets) if target == label]
        label_indices = sorted(
            label_indices,
            key=lambda idx: stable_path_score(paths[idx] if paths else str(idx), seed),
        )
        selected.extend(label_indices[:per_label])
    if len(selected) < max_samples:
        selected_set = set(selected)
        remaining = [idx for idx in range(len(targets)) if idx not in selected_set]
        remaining = sorted(
            remaining,
            key=lambda idx: stable_path_score(paths[idx] if paths else str(idx), seed + 17),
        )
        selected.extend(remaining[: max_samples - len(selected)])
    return selected[:max_samples]


def _stratified_indices(
    targets: list[int],
    test_fraction: float,
    seed: int,
    paths: list[str] | None = None,
) -> tuple[list[int], list[int]]:
    train_indices: list[int] = []
    test_indices: list[int] = []
    for label in sorted(set(targets)):
        label_indices = [idx for idx, target in enumerate(targets) if target == label]
        label_indices = sorted(
            label_indices,
            key=lambda idx: stable_path_score(paths[idx] if paths else str(idx), seed),
        )
        n_test = max(1, int(round(len(label_indices) * test_fraction)))
        test_indices.extend(label_indices[:n_test])
        train_indices.extend(label_indices[n_test:])
    return train_indices, test_indices


def _fake_class_index(class_to_idx: dict[str, int]) -> int:
    for class_name, idx in class_to_idx.items():
        if class_kind(class_name) == "fake":
            return idx
    raise ValueError(f"Could not identify generated/fake class in {class_to_idx}")


def _dataset_path(dataset, index: int) -> str:
    if isinstance(dataset, Subset):
        return _dataset_path(dataset.dataset, dataset.indices[index])
    return str(dataset.samples[index][0])


def make_datasets(args: argparse.Namespace, transform):
    layout = discover_layout(args.data_dir)
    if layout.train and layout.test:
        train_dataset = datasets.ImageFolder(layout.train, transform=transform)
        test_dataset = datasets.ImageFolder(layout.test, transform=transform)
    elif layout.single:
        train_base = datasets.ImageFolder(layout.single, transform=transform)
        test_base = datasets.ImageFolder(layout.single, transform=transform)
        train_indices, test_indices = _stratified_indices(
            _targets(train_base),
            args.val_fraction,
            args.seed,
            [path for path, _label in train_base.samples],
        )
        train_dataset = Subset(train_base, train_indices)
        test_dataset = Subset(test_base, test_indices)
    else:
        raise ValueError(f"Unsupported dataset layout: {layout}")

    train_targets = _targets(train_dataset.dataset) if isinstance(train_dataset, Subset) else _targets(train_dataset)
    test_targets = _targets(test_dataset.dataset) if isinstance(test_dataset, Subset) else _targets(test_dataset)
    if isinstance(train_dataset, Subset):
        train_paths = [train_dataset.dataset.samples[i][0] for i in train_dataset.indices]
        selected = _balanced_subset_indices(
            [train_targets[i] for i in train_dataset.indices],
            args.max_train_samples,
            args.seed,
            train_paths,
        )
        train_dataset = Subset(train_dataset.dataset, [train_dataset.indices[idx] for idx in selected])
    else:
        train_dataset = Subset(
            train_dataset,
            _balanced_subset_indices(
                train_targets,
                args.max_train_samples,
                args.seed,
                [path for path, _label in train_dataset.samples],
            ),
        )
    if isinstance(test_dataset, Subset):
        test_paths = [test_dataset.dataset.samples[i][0] for i in test_dataset.indices]
        selected = _balanced_subset_indices(
            [test_targets[i] for i in test_dataset.indices],
            args.max_test_samples,
            args.seed + 1,
            test_paths,
        )
        test_dataset = Subset(test_dataset.dataset, [test_dataset.indices[idx] for idx in selected])
    else:
        test_dataset = Subset(
            test_dataset,
            _balanced_subset_indices(
                test_targets,
                args.max_test_samples,
                args.seed + 1,
                [path for path, _label in test_dataset.samples],
            ),
        )

    base_dataset = train_dataset.dataset if isinstance(train_dataset, Subset) else train_dataset
    fake_idx = _fake_class_index(base_dataset.class_to_idx)
    return train_dataset, test_dataset, base_dataset.class_to_idx, fake_idx, layout


@torch.no_grad()
def _embedding_matrix(dataset, encoder, fake_idx: int, args: argparse.Namespace, desc: str):
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
    for images, batch_labels in tqdm(loader, desc=desc):
        images = images.to(device)
        embeddings = encode_batch(encoder, images).detach().cpu().numpy()
        labels_np = batch_labels.numpy()
        features.append(embeddings)
        labels.append((labels_np == fake_idx).astype(int))
        paths.extend(
            _dataset_path(loader.dataset, idx)
            for idx in range(dataset_offset, dataset_offset + len(labels_np))
        )
        dataset_offset += len(labels_np)
    if not features:
        raise ValueError(f"No embeddings extracted for {desc}")
    return np.vstack(features), np.concatenate(labels), paths


def _classifier(name: str, seed: int):
    if name == "logistic_regression":
        return Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "logreg",
                    LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed),
                ),
            ]
        )
    if name == "mlp":
        return Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "mlp",
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


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = ensure_dir(args.output_dir)
    device = resolve_device(args.device)
    spec = build_frozen_encoder(args.encoder, pretrained=args.pretrained)
    transform = frozen_encoder_transform(spec.image_size, spec.mean, spec.std)
    train_dataset, test_dataset, class_to_idx, fake_idx, layout = make_datasets(args, transform)

    x_train, y_train, _train_paths = _embedding_matrix(
        train_dataset, spec.model, fake_idx, args, "embeddings/train"
    )
    x_test, y_test, test_paths = _embedding_matrix(
        test_dataset, spec.model, fake_idx, args, "embeddings/test"
    )

    classifier = _classifier(args.classifier, args.seed)
    classifier.fit(x_train, y_train)
    if hasattr(classifier, "predict_proba"):
        scores = classifier.predict_proba(x_test)[:, 1]
    else:
        raw_scores = classifier.decision_function(x_test)
        scores = 1.0 / (1.0 + np.exp(-raw_scores))
    metrics = binary_metrics(y_test, scores)
    metrics.update(
        {
            "method": f"frozen_encoder_{args.encoder}_{args.classifier}",
            "encoder": args.encoder,
            "pretrained": bool(args.pretrained),
            "weights": spec.weights,
            "classifier": args.classifier,
            "embedding_dim": int(spec.embedding_dim),
            "image_size": int(spec.image_size),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "class_to_idx": class_to_idx,
            "fake_class_index": int(fake_idx),
            "device": str(device),
            "layout": {
                "train": str(layout.train) if layout.train else None,
                "test": str(layout.test) if layout.test else None,
                "single": str(layout.single) if layout.single else None,
            },
        }
    )
    write_json(metrics, output_dir / "metrics.json")
    joblib.dump(classifier, output_dir / "classifier.joblib")
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
    np.savez_compressed(
        output_dir / "embeddings.npz",
        x_train=x_train.astype(np.float32),
        y_train=y_train.astype(np.int64),
        x_test=x_test.astype(np.float32),
        y_test=y_test.astype(np.int64),
    )
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "y_true", "fake_score"])
        writer.writeheader()
        for path, truth, score in zip(test_paths, y_test, scores):
            writer.writerow({"path": str(path), "y_true": int(truth), "fake_score": float(score)})
    print(f"Wrote frozen encoder baseline results to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
