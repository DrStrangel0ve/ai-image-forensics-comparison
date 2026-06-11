from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from forensic_compare.catalog import load_catalog
from forensic_compare.datasets import class_kind, discover_layout, stable_path_score
from forensic_compare.metrics import binary_metrics
from forensic_compare.nn_model import build_model
from forensic_compare.utils import ensure_dir, read_json, resolve_device, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved neural model on a dataset.")
    parser.add_argument("--model-dir", required=True, help="Directory containing model.pt and metrics.json.")
    parser.add_argument("--target-key", default=None, help="Dataset key from configs/datasets.json.")
    parser.add_argument("--target-dir", default=None, help="Target dataset folder. Overrides --target-key.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default=None, help="Model architecture. Defaults to source metrics model.")
    parser.add_argument("--image-size", type=int, default=128)
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
def evaluate(model, loader, device, source_fake_idx: int, target_fake_idx: int):
    model.eval()
    y_true: list[int] = []
    y_score: list[float] = []
    rows: list[dict] = []
    dataset_offset = 0
    for images, labels in tqdm(loader, desc="eval"):
        images = images.to(device)
        logits = model(images)
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

    device = resolve_device(args.device)
    model = build_model(source_model_name, num_classes=2, pretrained=False).to(device)
    model.load_state_dict(torch.load(model_dir / "model.pt", map_location=device, weights_only=True))
    loader = DataLoader(
        target_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    metrics, rows = evaluate(model, loader, device, source_fake_idx, target_fake_idx)
    metrics.update(
        {
            "method": "cross_neural_net",
            "source_model_dir": str(model_dir),
            "target_dir": str(_target_dir(args)),
            "target_split": args.target_split,
            "model": source_model_name,
            "image_size": args.image_size,
            "device": str(device),
            "source_fake_idx": source_fake_idx,
            "target_fake_idx": target_fake_idx,
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
        writer.writerows(rows)
    print(f"Wrote cross-dataset neural evaluation to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
