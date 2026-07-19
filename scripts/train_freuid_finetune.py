from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forensic_compare.datasets import stable_path_score
from forensic_compare.freuid import freuid_competition_path, freuid_metrics
from forensic_compare.freuid_model import (
    build_freuid_model,
    required_freuid_input_size,
    supported_freuid_models,
)
from forensic_compare.freuid_transforms import build_document_transforms
from forensic_compare.metrics import binary_metrics
from forensic_compare.utils import ensure_dir, resolve_device, seed_everything, write_json


RESEARCH_TRACK = "post_freeze_highres_2026_07_13"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a document-aware FREUID fraud detector.")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--test-csv", default=None)
    parser.add_argument("--image-root", default="data/raw/freuid_2026/images")
    parser.add_argument("--output-dir", default="runs/freuid_finetune")
    parser.add_argument("--test-predictions-out", default=None)
    parser.add_argument("--model", choices=supported_freuid_models(), default="convnext_tiny")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--type-loss-weight", type=float, default=0.15)
    parser.add_argument("--focal-gamma", type=float, default=1.5)
    parser.add_argument("--hard-example-fraction", type=float, default=0.75)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--jpeg-augmentation-probability", type=float, default=0.25)
    parser.add_argument("--capture-augmentation-strength", type=float, default=0.0)
    parser.add_argument("--grid-rows", type=int, default=0)
    parser.add_argument("--grid-cols", type=int, default=0)
    parser.add_argument("--view-mode", choices=("auto", "single", "grid", "five_crop"), default="auto")
    parser.add_argument("--five-crop-zoom", type=float, default=1.15)
    parser.add_argument("--view-pooling", choices=("attention", "mean_logits"), default="attention")
    parser.add_argument("--view-chunk-size", type=int, default=0)
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--save-every-epoch", action="store_true")
    parser.add_argument("--type-adversarial-weight", type=float, default=0.0)
    parser.add_argument("--balanced-group-sampling", action="store_true")
    parser.add_argument("--forensic-residual", action="store_true")
    parser.add_argument("--selection-apcer-weight", type=float, default=0.25)
    parser.add_argument("--skip-errors", action="store_true")
    return parser.parse_args()


def _resolve_path(image_root: Path, raw_path: object) -> Path:
    raw = str(raw_path)
    candidates = [
        Path(raw),
        image_root / freuid_competition_path(raw),
        image_root / raw.replace("\\", "/"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve FREUID image {raw!r}; tried {candidates}")


def _public_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if "id" not in frame.columns:
        raise ValueError("Test CSV must contain an id column")
    result = frame.copy()
    result["id"] = result["id"].astype(str)
    if "image_path" not in result.columns:
        result["image_path"] = result["id"].map(lambda value: f"public_test/{value}.jpeg")
    return result


def _limit_frame(
    frame: pd.DataFrame,
    max_samples: int,
    seed: int,
    stratify: bool,
) -> pd.DataFrame:
    if max_samples <= 0 or max_samples >= len(frame):
        return frame.reset_index(drop=True)
    scored = frame.copy()
    scored["_score"] = scored["id"].astype(str).map(lambda value: stable_path_score(value, seed))
    if not stratify:
        return (
            scored.sort_values(["_score", "id"], kind="mergesort")
            .head(max_samples)
            .drop(columns=["_score"])
            .reset_index(drop=True)
        )
    group_columns = [column for column in ("type", "label") if column in scored.columns]
    groups = [
        group.sort_values(["_score", "id"], kind="mergesort")
        for _key, group in scored.groupby(group_columns, sort=True)
    ]
    selected = []
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


class FreuidCsvDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
        image_root: Path,
        transform,
        type_to_idx: dict[str, int],
        labeled: bool,
        skip_errors: bool,
    ) -> None:
        self.transform = transform
        self.type_to_idx = type_to_idx
        self.labeled = labeled
        self.rows: list[dict[str, object]] = []
        self.skipped: list[dict[str, str]] = []
        for row in frame.to_dict("records"):
            raw_path = row.get("local_path", row.get("image_path", ""))
            try:
                path = _resolve_path(image_root, raw_path)
                if skip_errors:
                    with Image.open(path) as image:
                        image.verify()
            except Exception as exc:
                if not skip_errors:
                    raise
                self.skipped.append({"id": str(row.get("id", "")), "error": repr(exc)})
                continue
            self.rows.append(
                {
                    "id": str(row["id"]),
                    "image_path": str(row.get("image_path", path.name)),
                    "local_path": str(path),
                    "label": int(row.get("label", 0)),
                    "type": str(row.get("type", "")),
                }
            )
        if not self.rows:
            raise ValueError("No usable FREUID images were found")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(str(row["local_path"])) as image:
            tensor = self.transform(image.convert("RGB"))
        label = float(row["label"]) if self.labeled else 0.0
        type_idx = self.type_to_idx.get(str(row["type"]), 0)
        return tensor, torch.tensor(label, dtype=torch.float32), torch.tensor(type_idx), str(row["id"])


def _fraud_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pos_weight: torch.Tensor,
    gamma: float,
    hard_fraction: float,
) -> torch.Tensor:
    losses = F.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight, reduction="none")
    if gamma > 0:
        probabilities = torch.sigmoid(logits)
        pt = torch.where(labels > 0.5, probabilities, 1.0 - probabilities)
        losses = losses * torch.pow(1.0 - pt, gamma)
    if 0 < hard_fraction < 1:
        keep = max(1, int(math.ceil(losses.numel() * hard_fraction)))
        losses = torch.topk(losses, k=keep, largest=True).values
    return losses.mean()


def _selection_objective(metrics: dict[str, float], apcer_weight: float) -> float:
    return float(metrics["audet_proxy"] + apcer_weight * metrics["apcer_at_1pct_bpcer"])


def _load_frames(args: argparse.Namespace):
    train = _limit_frame(pd.read_csv(args.train_csv), args.max_train_samples, args.seed, stratify=True)
    val = _limit_frame(pd.read_csv(args.val_csv), args.max_val_samples, args.seed + 1, stratify=True)
    test = None
    if args.test_csv:
        test = _limit_frame(
            _public_frame(pd.read_csv(args.test_csv)),
            args.max_test_samples,
            args.seed + 2,
            stratify=False,
        )
    type_names = sorted(set(train.get("type", pd.Series(dtype=str)).astype(str)) | set(val.get("type", pd.Series(dtype=str)).astype(str)))
    if not type_names:
        type_names = [""]
    return train, val, test, {name: index for index, name in enumerate(type_names)}


def _balanced_sampler(dataset: FreuidCsvDataset) -> WeightedRandomSampler:
    group_keys = [(str(row["type"]), int(row["label"])) for row in dataset.rows]
    counts = Counter(group_keys)
    weights = torch.as_tensor([1.0 / counts[key] for key in group_keys], dtype=torch.double)
    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[dict[str, float], list[dict[str, object]]]:
    model.eval()
    y_true: list[int] = []
    scores: list[float] = []
    ids: list[str] = []
    type_correct = 0
    n_types = 0
    for images, labels, type_indices, batch_ids in tqdm(loader, desc="freuid/eval", leave=False):
        images = images.to(device, non_blocking=True)
        fraud_logits, type_logits = model(images)
        probabilities = torch.sigmoid(fraud_logits).cpu().numpy()
        y_true.extend(labels.numpy().astype(int).tolist())
        scores.extend(probabilities.tolist())
        ids.extend(list(batch_ids))
        type_correct += int((type_logits.argmax(dim=1).cpu() == type_indices).sum().item())
        n_types += len(type_indices)
    metrics = binary_metrics(y_true, scores)
    metrics.update(freuid_metrics(y_true, scores))
    metrics["type_accuracy"] = type_correct / max(1, n_types)
    threshold = float(metrics["threshold_at_1pct_bpcer"])
    rows = [
        {"id": image_id, "y_true": truth, "fraud_score": score, "label": int(score >= threshold)}
        for image_id, truth, score in zip(ids, y_true, scores)
    ]
    return metrics, rows


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float,
) -> list[dict[str, object]]:
    model.eval()
    rows = []
    for images, _labels, _types, batch_ids in tqdm(loader, desc="freuid/test", leave=False):
        fraud_logits, _type_logits = model(images.to(device, non_blocking=True))
        scores = torch.sigmoid(fraud_logits).cpu().numpy()
        rows.extend(
            {
                "id": image_id,
                "fraud_score": float(score),
                "label": int(float(score) >= threshold),
            }
            for image_id, score in zip(batch_ids, scores)
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _resolve_view_mode(args: argparse.Namespace) -> str:
    has_grid = args.grid_rows > 0 and args.grid_cols > 0
    if (args.grid_rows > 0) != (args.grid_cols > 0):
        raise ValueError("--grid-rows and --grid-cols must either both be positive or both be zero")
    mode = str(args.view_mode)
    if mode == "auto":
        return "grid" if has_grid else "single"
    if mode == "grid" and not has_grid:
        raise ValueError("--view-mode grid requires positive --grid-rows and --grid-cols")
    if mode != "grid" and has_grid:
        raise ValueError("Grid dimensions may only be used with --view-mode grid or auto")
    return mode


def _view_count(view_mode: str, grid_rows: int, grid_cols: int) -> int:
    if view_mode == "five_crop":
        return 5
    if view_mode == "grid":
        return 1 + grid_rows * grid_cols
    return 1


def _checkpoint_payload(
    model: nn.Module,
    args: argparse.Namespace,
    metrics: dict[str, object],
    type_to_idx: dict[str, int],
    view_mode: str,
    multi_view: bool,
) -> dict[str, object]:
    return {
        "research_track": RESEARCH_TRACK,
        "competition_eligibility": "post_freeze_research_only",
        "model_state": model.state_dict(),
        "model": args.model,
        "multi_view": multi_view,
        "forensic_residual": bool(args.forensic_residual),
        "image_size": int(args.image_size),
        "view_mode": view_mode,
        "view_pooling": str(args.view_pooling),
        "view_chunk_size": int(args.view_chunk_size),
        "five_crop_zoom": float(args.five_crop_zoom),
        "grid_rows": int(args.grid_rows),
        "grid_cols": int(args.grid_cols),
        "freeze_encoder": bool(args.freeze_encoder),
        "lora_rank": int(args.lora_rank),
        "lora_alpha": float(args.lora_alpha),
        "type_to_idx": type_to_idx,
        "threshold": float(metrics["threshold_at_1pct_bpcer"]),
        "metrics": metrics,
    }


def _enable_gradient_checkpointing(model: nn.Module) -> None:
    encoder = model.encoder
    if isinstance(encoder, nn.Sequential) and isinstance(encoder[0], nn.Module):
        candidates = list(encoder.children())[::-1]
    else:
        candidates = [encoder]
    for candidate in candidates:
        setter = getattr(candidate, "set_grad_checkpointing", None)
        if callable(setter):
            setter(True)
            return
    raise ValueError("The selected encoder does not expose timm gradient checkpointing")


def run(args: argparse.Namespace) -> dict[str, object]:
    seed_everything(args.seed)
    output_dir = ensure_dir(args.output_dir)
    device = resolve_device(args.device)
    train_frame, val_frame, test_frame, type_to_idx = _load_frames(args)
    if args.gradient_accumulation_steps <= 0:
        raise ValueError("--gradient-accumulation-steps must be positive")
    required_size = required_freuid_input_size(args.model)
    if required_size is not None and args.image_size != required_size:
        raise ValueError(f"{args.model} requires --image-size {required_size}")
    view_mode = _resolve_view_mode(args)
    multi_view = view_mode != "single"
    train_transform, eval_transform = build_document_transforms(
        args.image_size,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        jpeg_probability=args.jpeg_augmentation_probability,
        capture_strength=args.capture_augmentation_strength,
        view_mode=view_mode,
        five_crop_zoom=args.five_crop_zoom,
    )
    image_root = Path(args.image_root)
    train_dataset = FreuidCsvDataset(train_frame, image_root, train_transform, type_to_idx, True, args.skip_errors)
    val_dataset = FreuidCsvDataset(val_frame, image_root, eval_transform, type_to_idx, True, args.skip_errors)
    test_dataset = (
        FreuidCsvDataset(test_frame, image_root, eval_transform, type_to_idx, False, args.skip_errors)
        if test_frame is not None
        else None
    )
    loader_kwargs = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    train_sampler = _balanced_sampler(train_dataset) if args.balanced_group_sampling else None
    train_loader = DataLoader(
        train_dataset,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs) if test_dataset is not None else None

    model = build_freuid_model(
        args.model,
        num_types=len(type_to_idx),
        pretrained=args.pretrained,
        dropout=args.dropout,
        multi_view=multi_view,
        forensic_residual=args.forensic_residual,
        view_pooling=args.view_pooling,
        freeze_encoder=args.freeze_encoder,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        view_chunk_size=args.view_chunk_size,
    ).to(device)
    if args.gradient_checkpointing:
        _enable_gradient_checkpointing(model)
    if device.type == "cuda" and not multi_view:
        model = model.to(memory_format=torch.channels_last)
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("The model has no trainable parameters")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    positives = float(train_frame["label"].sum())
    negatives = float(len(train_frame) - positives)
    pos_weight = torch.tensor(negatives / max(1.0, positives), dtype=torch.float32, device=device)

    history = []
    best_objective = float("inf")
    checkpoint_path = output_dir / "model.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        if args.freeze_encoder or args.lora_rank > 0:
            model.encoder.eval()
        running_loss = 0.0
        seen = 0
        optimizer.zero_grad(set_to_none=True)
        for batch_index, (images, labels, type_indices, _ids) in enumerate(
            tqdm(train_loader, desc=f"freuid/train/{epoch}", leave=False)
        ):
            images = images.to(device, non_blocking=True)
            if device.type == "cuda" and images.ndim == 4:
                images = images.to(memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)
            type_indices = type_indices.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                fraud_logits, type_logits = model(
                    images,
                    type_adversarial_scale=args.type_adversarial_weight,
                )
                fraud_loss = _fraud_loss(
                    fraud_logits,
                    labels,
                    pos_weight,
                    gamma=args.focal_gamma,
                    hard_fraction=args.hard_example_fraction,
                )
                type_loss = F.cross_entropy(type_logits, type_indices)
                loss = fraud_loss + args.type_loss_weight * type_loss
            group_start = (batch_index // args.gradient_accumulation_steps) * args.gradient_accumulation_steps
            group_size = min(args.gradient_accumulation_steps, len(train_loader) - group_start)
            scaled_loss = loss / group_size
            scaler.scale(scaled_loss).backward()
            should_step = (
                (batch_index + 1) % args.gradient_accumulation_steps == 0
                or batch_index + 1 == len(train_loader)
            )
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running_loss += float(loss.item()) * len(labels)
            seen += len(labels)
        metrics, rows = evaluate(model, val_loader, device)
        metrics.update(
            {
                "epoch": epoch,
                "train_loss": running_loss / max(1, seen),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )
        metrics["selection_objective"] = _selection_objective(metrics, args.selection_apcer_weight)
        history.append(metrics)
        _write_csv(output_dir / f"val_predictions_epoch{epoch}.csv", rows, ["id", "y_true", "fraud_score", "label"])
        print(
            f"epoch={epoch} objective={metrics['selection_objective']:.6f} "
            f"audet={metrics['audet_proxy']:.6f} apcer={metrics['apcer_at_1pct_bpcer']:.6f} "
            f"auc={metrics['roc_auc']:.6f}"
        )
        payload = _checkpoint_payload(model, args, metrics, type_to_idx, view_mode, multi_view)
        if args.save_every_epoch:
            torch.save(payload, output_dir / f"model_epoch{epoch}.pt")
        if metrics["selection_objective"] < best_objective:
            best_objective = float(metrics["selection_objective"])
            torch.save(payload, checkpoint_path)
        scheduler.step()

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    best_metrics, best_rows = evaluate(model, val_loader, device)
    best_metrics["selection_objective"] = _selection_objective(best_metrics, args.selection_apcer_weight)
    best_metrics.update(
        {
            "research_track": RESEARCH_TRACK,
            "competition_eligibility": "post_freeze_research_only",
            "method": f"freuid_finetune_{args.model}_multitask",
            "model": args.model,
            "pretrained": bool(args.pretrained),
            "image_size": int(args.image_size),
            "multi_view": multi_view,
            "forensic_residual": bool(args.forensic_residual),
            "view_mode": view_mode,
            "view_pooling": str(args.view_pooling),
            "view_chunk_size": int(args.view_chunk_size),
            "five_crop_zoom": float(args.five_crop_zoom),
            "n_views": _view_count(view_mode, int(args.grid_rows), int(args.grid_cols)),
            "grid_rows": int(args.grid_rows),
            "grid_cols": int(args.grid_cols),
            "parameters": {
                "total": int(sum(parameter.numel() for parameter in model.parameters())),
                "trainable": int(sum(parameter.numel() for parameter in trainable_parameters)),
            },
            "device": str(device),
            "n_train": len(train_dataset),
            "n_val": len(val_dataset),
            "n_train_skipped": len(train_dataset.skipped),
            "n_val_skipped": len(val_dataset.skipped),
            "type_to_idx": type_to_idx,
            "selected_epoch": int(checkpoint["metrics"]["epoch"]),
            "threshold_at_1pct_bpcer": float(checkpoint["threshold"]),
            "training": {
                "lr": float(args.lr),
                "weight_decay": float(args.weight_decay),
                "focal_gamma": float(args.focal_gamma),
                "hard_example_fraction": float(args.hard_example_fraction),
                "type_loss_weight": float(args.type_loss_weight),
                "jpeg_augmentation_probability": float(args.jpeg_augmentation_probability),
                "capture_augmentation_strength": float(args.capture_augmentation_strength),
                "type_adversarial_weight": float(args.type_adversarial_weight),
                "balanced_group_sampling": bool(args.balanced_group_sampling),
                "forensic_residual": bool(args.forensic_residual),
                "freeze_encoder": bool(args.freeze_encoder),
                "lora_rank": int(args.lora_rank),
                "lora_alpha": float(args.lora_alpha),
                "gradient_checkpointing": bool(args.gradient_checkpointing),
                "gradient_accumulation_steps": int(args.gradient_accumulation_steps),
                "save_every_epoch": bool(args.save_every_epoch),
            },
        }
    )
    _write_csv(output_dir / "val_predictions.csv", best_rows, ["id", "y_true", "fraud_score", "label"])
    write_json(history, output_dir / "history.json")
    write_json(best_metrics, output_dir / "metrics.json")
    write_json(
        {"train": train_dataset.skipped, "val": val_dataset.skipped, "test": test_dataset.skipped if test_dataset else []},
        output_dir / "skipped.json",
    )

    if test_loader is not None:
        test_rows = predict(model, test_loader, device, threshold=float(checkpoint["threshold"]))
        test_path = Path(args.test_predictions_out) if args.test_predictions_out else output_dir / "test_predictions.csv"
        _write_csv(test_path, test_rows, ["id", "fraud_score", "label"])
        best_metrics.update(
            {
                "n_test": len(test_dataset),
                "n_test_skipped": len(test_dataset.skipped),
                "test_predictions_path": str(test_path),
            }
        )
        write_json(best_metrics, output_dir / "metrics.json")
    return best_metrics


def main() -> None:
    metrics = run(parse_args())
    print(Path(str(metrics.get("test_predictions_path", ""))).resolve() if metrics.get("test_predictions_path") else "")
    print(
        f"audet_proxy={metrics['audet_proxy']:.6f} "
        f"apcer_at_1pct_bpcer={metrics['apcer_at_1pct_bpcer']:.6f}"
    )


if __name__ == "__main__":
    main()
