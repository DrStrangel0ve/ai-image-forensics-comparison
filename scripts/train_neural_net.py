from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from forensic_compare.datasets import class_kind, discover_layout
from forensic_compare.datasets import stable_path_score
from forensic_compare.metrics import binary_metrics
from forensic_compare.nn_model import build_model
from forensic_compare.utils import ensure_dir, resolve_device, seed_everything, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a neural baseline for real-vs-generated detection.")
    parser.add_argument("--data-dir", default="data/raw/cifake")
    parser.add_argument("--output-dir", default="runs/resnet18")
    parser.add_argument("--model", default="resnet18", choices=["resnet18", "tiny_cnn"])
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet weights where available.")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=96)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
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


def make_datasets(args: argparse.Namespace):
    train_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    layout = discover_layout(args.data_dir)
    if layout.train and layout.test:
        train_dataset = datasets.ImageFolder(layout.train, transform=train_transform)
        test_dataset = datasets.ImageFolder(layout.test, transform=eval_transform)
    elif layout.single:
        train_base = datasets.ImageFolder(layout.single, transform=train_transform)
        test_base = datasets.ImageFolder(layout.single, transform=eval_transform)
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
        train_dataset = Subset(
            train_dataset.dataset,
            [train_dataset.indices[idx] for idx in _balanced_subset_indices(
                [train_targets[i] for i in train_dataset.indices],
                args.max_train_samples,
                args.seed,
                train_paths,
            )],
        )
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
        test_dataset = Subset(
            test_dataset.dataset,
            [test_dataset.indices[idx] for idx in _balanced_subset_indices(
                [test_targets[i] for i in test_dataset.indices],
                args.max_test_samples,
                args.seed + 1,
                test_paths,
            )],
        )
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


def train_one_epoch(model, loader, criterion, optimizer, device, scaler) -> float:
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += float(loss.item()) * images.size(0)
    return running_loss / max(1, len(loader.dataset))


@torch.no_grad()
def evaluate(model, loader, device, fake_idx: int) -> tuple[dict, list[dict]]:
    model.eval()
    y_true: list[int] = []
    y_score: list[float] = []
    rows: list[dict] = []
    dataset_offset = 0
    for images, labels in tqdm(loader, desc="eval", leave=False):
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, fake_idx].detach().cpu().numpy()
        labels_np = labels.numpy()
        fake_truth = (labels_np == fake_idx).astype(int)
        y_true.extend(fake_truth.tolist())
        y_score.extend(probs.tolist())
        paths = [
            _dataset_path(loader.dataset, idx)
            for idx in range(dataset_offset, dataset_offset + len(labels_np))
        ]
        dataset_offset += len(labels_np)
        rows.extend(
            {"path": path, "y_true": int(truth), "fake_score": float(score)}
            for path, truth, score in zip(paths, fake_truth, probs)
        )
    return binary_metrics(y_true, y_score), rows


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    output_dir = ensure_dir(args.output_dir)
    train_dataset, test_dataset, class_to_idx, fake_idx, layout = make_datasets(args)
    device = resolve_device(args.device)

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    model = build_model(args.model, num_classes=len(class_to_idx), pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    history = []
    best_accuracy = -1.0
    best_epoch = 0
    best_model_path = output_dir / "model.pt"
    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
        metrics, _ = evaluate(model, test_loader, device, fake_idx)
        metrics["epoch"] = epoch
        metrics["train_loss"] = loss
        history.append(metrics)
        print(f"epoch={epoch} loss={loss:.4f} accuracy={metrics['accuracy']:.4f} f1={metrics['f1']:.4f}")
        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            best_epoch = epoch
            torch.save(model.state_dict(), best_model_path)

    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    metrics, rows = evaluate(model, test_loader, device, fake_idx)
    metrics.update(
        {
            "method": "neural_net",
            "model": args.model,
            "device": str(device),
            "class_to_idx": class_to_idx,
            "fake_idx": fake_idx,
            "layout": {
                "train": str(layout.train) if layout.train else None,
                "test": str(layout.test) if layout.test else None,
                "single": str(layout.single) if layout.single else None,
            },
            "n_train": len(train_dataset),
            "n_test": len(test_dataset),
            "selected_epoch": best_epoch,
            "best_accuracy": best_accuracy,
        }
    )
    write_json(metrics, output_dir / "metrics.json")
    write_json(history, output_dir / "history.json")
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "y_true", "fake_score"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote neural-net results to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
