from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

from forensic_compare.conventional import extract_feature_set, feature_names
from forensic_compare.metrics import binary_metrics
from forensic_compare.nn_model import build_feature_fusion_model
from forensic_compare.utils import ensure_dir, resolve_device, seed_everything, write_json
from scripts.train_neural_net import _dataset_path, make_datasets


FEATURE_SETS = [
    "photometric",
    "noise",
    "noise_v2",
    "noise_v3",
    "noise_v4",
    "reconstruction_lite",
    "combined",
    "combined_v2",
    "combined_v3",
    "combined_v4",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a neural baseline fused with physics/signal forensic features."
    )
    parser.add_argument("--data-dir", default="data/raw/cifake")
    parser.add_argument("--output-dir", default="runs/physics_guided_resnet18")
    parser.add_argument("--model", default="resnet18", choices=["resnet18", "tiny_cnn"])
    parser.add_argument("--pretrained", action="store_true", help="Use ImageNet weights where available.")
    parser.add_argument("--physics-feature-set", choices=FEATURE_SETS, default="combined_v3")
    parser.add_argument("--feature-image-size", type=int, default=128)
    parser.add_argument("--feature-hidden-dim", type=int, default=128)
    parser.add_argument("--fusion-dropout", type=float, default=0.2)
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
    parser.add_argument("--skip-errors", action="store_true", help="Skip unreadable feature rows.")
    parser.set_defaults(train_augment_variants=[])
    return parser.parse_args()


class PhysicsFeatureDataset(Dataset):
    def __init__(self, image_dataset, features: np.ndarray) -> None:
        self.image_dataset = image_dataset
        self.features = torch.from_numpy(features.astype(np.float32, copy=False))

    def __len__(self) -> int:
        return len(self.image_dataset)

    def __getitem__(self, index: int):
        image, label = self.image_dataset[index]
        return image, self.features[index], label


def _extract_feature_matrix(
    dataset,
    feature_set: str,
    image_size: int,
    desc: str,
    skip_errors: bool,
) -> tuple[np.ndarray, list[str], list[dict], list[int]]:
    features = []
    paths = []
    skipped = []
    kept_indices = []
    for index in tqdm(range(len(dataset)), desc=desc):
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
        raise ValueError(f"No usable feature rows for {desc}")
    return np.vstack(features).astype(np.float32), paths, skipped, kept_indices


def _standardize_features(
    train_features: np.ndarray,
    test_features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = train_features.mean(axis=0, keepdims=True)
    std = train_features.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (
        ((train_features - mean) / std).astype(np.float32),
        ((test_features - mean) / std).astype(np.float32),
        mean.reshape(-1).astype(np.float32),
        std.reshape(-1).astype(np.float32),
    )


def train_one_epoch(model, loader, criterion, optimizer, device, scaler) -> float:
    model.train()
    running_loss = 0.0
    for images, feature_vectors, labels in tqdm(loader, desc="train", leave=False):
        images = images.to(device)
        feature_vectors = feature_vectors.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            logits = model(images, feature_vectors)
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
    for images, feature_vectors, labels in tqdm(loader, desc="eval", leave=False):
        images = images.to(device)
        feature_vectors = feature_vectors.to(device)
        logits = model(images, feature_vectors)
        probs = torch.softmax(logits, dim=1)[:, fake_idx].detach().cpu().numpy()
        labels_np = labels.numpy()
        fake_truth = (labels_np == fake_idx).astype(int)
        y_true.extend(fake_truth.tolist())
        y_score.extend(probs.tolist())
        paths = [
            _dataset_path(loader.dataset.image_dataset, idx)
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
    train_image_dataset, test_image_dataset, class_to_idx, fake_idx, layout = make_datasets(args)
    train_raw_features, _train_paths, skipped_train, train_keep_indices = _extract_feature_matrix(
        train_image_dataset,
        args.physics_feature_set,
        args.feature_image_size,
        "physics-features/train",
        args.skip_errors,
    )
    test_raw_features, _test_paths, skipped_test, test_keep_indices = _extract_feature_matrix(
        test_image_dataset,
        args.physics_feature_set,
        args.feature_image_size,
        "physics-features/test",
        args.skip_errors,
    )
    if len(train_keep_indices) != len(train_image_dataset):
        train_image_dataset = Subset(train_image_dataset, train_keep_indices)
    if len(test_keep_indices) != len(test_image_dataset):
        test_image_dataset = Subset(test_image_dataset, test_keep_indices)
    train_features, test_features, feature_mean, feature_std = _standardize_features(
        train_raw_features, test_raw_features
    )
    train_dataset = PhysicsFeatureDataset(train_image_dataset, train_features)
    test_dataset = PhysicsFeatureDataset(test_image_dataset, test_features)
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
    model = build_feature_fusion_model(
        args.model,
        feature_dim=train_features.shape[1],
        num_classes=len(class_to_idx),
        pretrained=args.pretrained,
        hidden_dim=args.feature_hidden_dim,
        dropout=args.fusion_dropout,
    ).to(device)
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
    names = feature_names(args.physics_feature_set)
    metrics.update(
        {
            "method": "physics_guided_net",
            "model": args.model,
            "physics_feature_set": args.physics_feature_set,
            "feature_names": names,
            "feature_mean": feature_mean.tolist(),
            "feature_std": feature_std.tolist(),
            "feature_hidden_dim": args.feature_hidden_dim,
            "fusion_dropout": args.fusion_dropout,
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
            "n_skipped_train": len(skipped_train),
            "n_skipped_test": len(skipped_test),
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
    print(f"Wrote physics-guided neural results to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
