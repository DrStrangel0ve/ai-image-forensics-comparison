from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from forensic_compare.catalog import load_catalog
from forensic_compare.conventional import extract_feature_set
from forensic_compare.datasets import class_kind, discover_layout, stable_path_score
from forensic_compare.metrics import binary_metrics
from forensic_compare.nn_model import build_feature_fusion_model
from forensic_compare.utils import ensure_dir, read_json, resolve_device, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved physics-guided fusion model.")
    parser.add_argument("--model-dir", required=True, help="Directory containing model.pt and metrics.json.")
    parser.add_argument("--target-key", default=None, help="Dataset key from configs/datasets.json.")
    parser.add_argument("--target-dir", default=None, help="Target dataset folder. Overrides --target-key.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default=None, help="Image model architecture. Defaults to source metrics model.")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--feature-image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--target-split", choices=["all", "test"], default="all")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--skip-errors", action="store_true")
    return parser.parse_args()


class FeatureFusionEvalDataset(Dataset):
    def __init__(self, image_dataset, features: np.ndarray) -> None:
        self.image_dataset = image_dataset
        self.features = torch.from_numpy(features.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return len(self.image_dataset)

    def __getitem__(self, index: int):
        image, label = self.image_dataset[index]
        return image, self.features[index], label


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
        return _class_to_idx(dataset.dataset)
    if isinstance(dataset, ConcatDataset):
        return _class_to_idx(dataset.datasets[0])
    return dataset.class_to_idx


def _dataset_path(dataset, index: int) -> str:
    if isinstance(dataset, FeatureFusionEvalDataset):
        return _dataset_path(dataset.image_dataset, index)
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


def _extract_feature_matrix(
    dataset,
    feature_set: str,
    image_size: int,
    skip_errors: bool,
) -> tuple[np.ndarray, list[str], list[dict], list[int]]:
    features = []
    paths = []
    skipped = []
    kept_indices = []
    for index in tqdm(range(len(dataset)), desc="physics-features/target"):
        path = _dataset_path(dataset, index)
        try:
            row = extract_feature_set(path, image_size=image_size, feature_set=feature_set)
        except Exception as exc:
            if not skip_errors:
                raise
            skipped.append({"path": path, "error": repr(exc)})
            continue
        features.append(row)
        paths.append(path)
        kept_indices.append(index)
    if not features:
        raise ValueError("No usable target feature rows")
    return np.vstack(features).astype(np.float32), paths, skipped, kept_indices


def _standardize_target_features(
    target_features: np.ndarray,
    feature_mean: list[float],
    feature_std: list[float],
) -> np.ndarray:
    mean = np.asarray(feature_mean, dtype=np.float32).reshape(1, -1)
    std = np.asarray(feature_std, dtype=np.float32).reshape(1, -1)
    if target_features.shape[1] != mean.shape[1]:
        raise ValueError(f"Feature dimension mismatch: target={target_features.shape[1]} model={mean.shape[1]}")
    std = np.where(std < 1e-6, 1.0, std)
    return ((target_features - mean) / std).astype(np.float32)


@torch.no_grad()
def evaluate(model, loader, device, source_fake_idx: int, target_fake_idx: int):
    model.eval()
    y_true: list[int] = []
    y_score: list[float] = []
    rows: list[dict] = []
    dataset_offset = 0
    for images, feature_vectors, labels in tqdm(loader, desc="eval"):
        images = images.to(device)
        feature_vectors = feature_vectors.to(device)
        logits = model(images, feature_vectors)
        probs = torch.softmax(logits, dim=1)[:, source_fake_idx].detach().cpu().numpy()
        labels_np = labels.numpy()
        fake_truth = (labels_np == target_fake_idx).astype(int)
        paths = [
            _dataset_path(loader.dataset, index)
            for index in range(dataset_offset, dataset_offset + len(labels_np))
        ]
        dataset_offset += len(labels_np)
        y_true.extend(fake_truth.tolist())
        y_score.extend(probs.tolist())
        rows.extend(
            {"path": path, "y_true": int(truth), "fake_score": float(score)}
            for path, truth, score in zip(paths, fake_truth, probs)
        )
    return binary_metrics(y_true, y_score), rows


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    output_dir = ensure_dir(args.output_dir)
    source_metrics = read_json(model_dir / "metrics.json")
    source_model_name = args.model or source_metrics["model"]
    source_fake_idx = int(source_metrics["fake_idx"])
    physics_feature_set = source_metrics["physics_feature_set"]
    feature_mean = source_metrics["feature_mean"]
    feature_std = source_metrics["feature_std"]
    feature_hidden_dim = int(source_metrics.get("feature_hidden_dim", 128))
    fusion_dropout = float(source_metrics.get("fusion_dropout", 0.2))

    transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    target_dataset, layout = _dataset(_target_dir(args), transform, args)
    target_class_to_idx = _class_to_idx(target_dataset)
    target_fake_idx = _fake_class_index(target_class_to_idx)
    target_raw_features, _paths, skipped, kept_indices = _extract_feature_matrix(
        target_dataset,
        physics_feature_set,
        args.feature_image_size,
        args.skip_errors,
    )
    if len(kept_indices) != len(target_dataset):
        target_dataset = Subset(target_dataset, kept_indices)
    target_features = _standardize_target_features(target_raw_features, feature_mean, feature_std)
    eval_dataset = FeatureFusionEvalDataset(target_dataset, target_features)

    device = resolve_device(args.device)
    model = build_feature_fusion_model(
        source_model_name,
        feature_dim=target_features.shape[1],
        num_classes=len(source_metrics["class_to_idx"]),
        pretrained=False,
        hidden_dim=feature_hidden_dim,
        dropout=fusion_dropout,
    ).to(device)
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location=device, weights_only=True))
    loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    metrics, rows = evaluate(model, loader, device, source_fake_idx, target_fake_idx)
    metrics.update(
        {
            "method": "cross_physics_guided_net",
            "source_model_dir": str(model_dir),
            "target_dir": str(_target_dir(args)),
            "target_split": args.target_split,
            "model": source_model_name,
            "physics_feature_set": physics_feature_set,
            "image_size": args.image_size,
            "feature_image_size": args.feature_image_size,
            "feature_hidden_dim": feature_hidden_dim,
            "fusion_dropout": fusion_dropout,
            "device": str(device),
            "source_fake_idx": source_fake_idx,
            "target_fake_idx": target_fake_idx,
            "target_class_to_idx": target_class_to_idx,
            "n_target": len(eval_dataset),
            "n_skipped_target": len(skipped),
            "layout": {
                "train": str(layout.train) if layout.train else None,
                "test": str(layout.test) if layout.test else None,
                "single": str(layout.single) if layout.single else None,
            },
        }
    )
    write_json(metrics, output_dir / "metrics.json")
    if skipped:
        write_json(skipped, output_dir / "skipped.json")
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "y_true", "fake_score"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote physics-guided evaluation to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
