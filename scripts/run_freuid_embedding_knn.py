from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forensic_compare.freuid import freuid_competition_path, freuid_metrics
from forensic_compare.freuid_model import build_freuid_model
from forensic_compare.metrics import binary_metrics
from forensic_compare.utils import ensure_dir, resolve_device, seed_everything, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Template-aware k-NN scoring from a frozen FREUID encoder.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--test-csv", default=None)
    parser.add_argument("--image-root", default="data/raw/freuid_2026/images")
    parser.add_argument("--output-dir", default="runs/freuid_embedding_knn")
    parser.add_argument("--test-predictions-out", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--query-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=6)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 3, 5, 9, 15, 31])
    parser.add_argument("--temperatures", nargs="+", type=float, default=[10.0, 25.0, 50.0])
    parser.add_argument("--head-weights", nargs="+", type=float, default=[0.0, 0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--selection-apcer-weight", type=float, default=0.25)
    return parser.parse_args()


class Letterbox:
    def __init__(self, size: int) -> None:
        self.size = int(size)

    def __call__(self, image: Image.Image) -> Image.Image:
        return ImageOps.pad(
            image,
            (self.size, self.size),
            method=Image.Resampling.BICUBIC,
            color=(127, 127, 127),
        )


def _resolve_path(image_root: Path, row: dict[str, object]) -> Path:
    raw = str(row.get("local_path", row.get("image_path", "")))
    candidates = [
        Path(raw),
        image_root / freuid_competition_path(raw),
        image_root / raw.replace("\\", "/"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve {raw!r}; tried {candidates}")


class CsvImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, image_root: Path, image_size: int, labeled: bool) -> None:
        self.rows = []
        self.labeled = labeled
        self.transform = transforms.Compose(
            [
                Letterbox(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            ]
        )
        for row in frame.to_dict("records"):
            path = _resolve_path(image_root, row)
            self.rows.append(
                {
                    "id": str(row["id"]),
                    "path": str(path),
                    "label": int(row.get("label", 0)),
                    "type": str(row.get("type", "")),
                }
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(str(row["path"])) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, int(row["label"]), str(row["id"]), str(row["type"])


def _frame(path: str | Path, labeled: bool) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame["id"] = frame["id"].astype(str)
    if "image_path" not in frame.columns and "local_path" not in frame.columns:
        split = "train" if labeled else "public_test"
        frame["image_path"] = frame["id"].map(lambda value: f"{split}/{value}.jpeg")
    if labeled and "label" not in frame.columns:
        raise ValueError(f"{path} is missing label")
    return frame


@torch.no_grad()
def _extract(
    model,
    dataset: CsvImageDataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    desc: str,
) -> dict[str, object]:
    loader = DataLoader(
        dataset,
        batch_size=max(1, batch_size),
        shuffle=False,
        num_workers=max(0, num_workers),
        pin_memory=device.type == "cuda",
    )
    embeddings = []
    head_scores = []
    type_indices = []
    labels = []
    ids = []
    types = []
    model.eval()
    for images, batch_labels, batch_ids, batch_types in tqdm(loader, desc=desc):
        images = images.to(device, non_blocking=True)
        if device.type == "cuda":
            images = images.to(memory_format=torch.channels_last)
        encoded = model.encoder(images)
        if encoded.ndim > 2:
            encoded = torch.flatten(encoded, 1)
        fraud_logits = model.fraud_head(encoded).squeeze(1)
        type_logits = model.type_head(encoded)
        embeddings.append(F.normalize(encoded, dim=1).cpu().numpy().astype(np.float32))
        head_scores.extend(torch.sigmoid(fraud_logits).cpu().numpy().astype(float).tolist())
        type_indices.extend(type_logits.argmax(dim=1).cpu().numpy().astype(int).tolist())
        labels.extend(batch_labels.numpy().astype(int).tolist())
        ids.extend(list(batch_ids))
        types.extend(list(batch_types))
    return {
        "embeddings": np.vstack(embeddings),
        "head_scores": np.asarray(head_scores, dtype=np.float64),
        "type_indices": np.asarray(type_indices, dtype=np.int64),
        "labels": np.asarray(labels, dtype=np.int64),
        "ids": ids,
        "types": types,
    }


@torch.no_grad()
def _knn_scores(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    query_embeddings: np.ndarray,
    k_values: list[int],
    temperatures: list[float],
    query_batch_size: int,
    device: torch.device,
) -> dict[tuple[int, float], np.ndarray]:
    max_k = min(max(k_values), len(train_embeddings))
    train_tensor = torch.from_numpy(train_embeddings).to(device)
    label_tensor = torch.from_numpy(train_labels.astype(np.float32)).to(device)
    if device.type == "cuda":
        train_tensor = train_tensor.to(dtype=torch.float16)
    outputs = {(k, temperature): [] for k in k_values for temperature in temperatures}
    for start in tqdm(range(0, len(query_embeddings), query_batch_size), desc="freuid/knn"):
        query = torch.from_numpy(query_embeddings[start : start + query_batch_size]).to(device)
        if device.type == "cuda":
            query = query.to(dtype=torch.float16)
        similarities = query @ train_tensor.T
        top_similarities, top_indices = torch.topk(similarities, k=max_k, dim=1)
        top_labels = label_tensor[top_indices]
        for k in k_values:
            effective_k = min(k, max_k)
            for temperature in temperatures:
                weights = torch.softmax(top_similarities[:, :effective_k].float() * temperature, dim=1)
                scores = torch.sum(weights * top_labels[:, :effective_k], dim=1)
                outputs[(k, temperature)].append(scores.cpu().numpy().astype(np.float64))
    return {key: np.concatenate(parts) for key, parts in outputs.items()}


def _metric_row(
    y_true: np.ndarray,
    scores: np.ndarray,
    k: int,
    temperature: float,
    head_weight: float,
    apcer_weight: float,
) -> dict[str, float | int]:
    metrics = binary_metrics(y_true, scores)
    metrics.update(freuid_metrics(y_true, scores))
    metrics.update(
        {
            "k": int(k),
            "temperature": float(temperature),
            "head_weight": float(head_weight),
            "selection_objective": float(
                metrics["audet_proxy"] + apcer_weight * metrics["apcer_at_1pct_bpcer"]
            ),
        }
    )
    return metrics


def run(args: argparse.Namespace) -> dict[str, object]:
    seed_everything(args.seed)
    output_dir = ensure_dir(args.output_dir)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    type_to_idx = dict(checkpoint["type_to_idx"])
    model = build_freuid_model(
        str(checkpoint["model"]),
        num_types=len(type_to_idx),
        pretrained=False,
        multi_view=bool(checkpoint.get("multi_view", False)),
        forensic_residual=bool(checkpoint.get("forensic_residual", False)),
        view_pooling=str(checkpoint.get("view_pooling", "attention")),
        freeze_encoder=bool(checkpoint.get("freeze_encoder", False)),
        lora_rank=int(checkpoint.get("lora_rank", 0)),
        lora_alpha=float(checkpoint.get("lora_alpha", 16.0)),
        view_chunk_size=int(checkpoint.get("view_chunk_size", 0)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device).eval()
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    train_data = _extract(
        model,
        CsvImageDataset(_frame(args.train_csv, True), Path(args.image_root), int(checkpoint["image_size"]), True),
        device,
        args.batch_size,
        args.num_workers,
        "freuid/embed/train",
    )
    val_data = _extract(
        model,
        CsvImageDataset(_frame(args.val_csv, True), Path(args.image_root), int(checkpoint["image_size"]), True),
        device,
        args.batch_size,
        args.num_workers,
        "freuid/embed/val",
    )
    val_knn = _knn_scores(
        train_data["embeddings"],
        train_data["labels"],
        val_data["embeddings"],
        args.k_values,
        args.temperatures,
        args.query_batch_size,
        device,
    )
    rows = []
    for (k, temperature), knn_scores in val_knn.items():
        for head_weight in args.head_weights:
            scores = head_weight * val_data["head_scores"] + (1.0 - head_weight) * knn_scores
            rows.append(
                _metric_row(
                    val_data["labels"],
                    scores,
                    k,
                    temperature,
                    head_weight,
                    args.selection_apcer_weight,
                )
            )
    rows.sort(key=lambda row: (row["selection_objective"], row["apcer_at_1pct_bpcer"], row["audet_proxy"]))
    best = rows[0]
    pd.DataFrame(rows).to_csv(output_dir / "knn_grid.csv", index=False)

    best_knn = val_knn[(int(best["k"]), float(best["temperature"]))]
    best_val_scores = float(best["head_weight"]) * val_data["head_scores"] + (1.0 - float(best["head_weight"])) * best_knn
    threshold = float(best["threshold_at_1pct_bpcer"])
    _write_predictions(
        output_dir / "val_predictions.csv",
        val_data["ids"],
        best_val_scores,
        threshold,
        val_data["labels"],
    )
    np.savez_compressed(
        output_dir / "embeddings.npz",
        train_embeddings=train_data["embeddings"].astype(np.float16),
        train_labels=train_data["labels"],
        val_embeddings=val_data["embeddings"].astype(np.float16),
        val_labels=val_data["labels"],
    )

    summary: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "n_train": len(train_data["ids"]),
        "n_val": len(val_data["ids"]),
        "embedding_dim": int(train_data["embeddings"].shape[1]),
        "best": best,
        "top_rows": rows[:10],
        "type_to_idx": type_to_idx,
    }
    if args.test_csv:
        test_data = _extract(
            model,
            CsvImageDataset(_frame(args.test_csv, False), Path(args.image_root), int(checkpoint["image_size"]), False),
            device,
            args.batch_size,
            args.num_workers,
            "freuid/embed/test",
        )
        test_knn = _knn_scores(
            train_data["embeddings"],
            train_data["labels"],
            test_data["embeddings"],
            [int(best["k"])],
            [float(best["temperature"])],
            args.query_batch_size,
            device,
        )[(int(best["k"]), float(best["temperature"]))]
        test_scores = float(best["head_weight"]) * test_data["head_scores"] + (1.0 - float(best["head_weight"])) * test_knn
        test_path = Path(args.test_predictions_out) if args.test_predictions_out else output_dir / "test_predictions.csv"
        _write_predictions(test_path, test_data["ids"], test_scores, threshold)
        diagnostics = pd.DataFrame(
            {
                "id": test_data["ids"],
                "predicted_type_index": test_data["type_indices"],
                "head_score": test_data["head_scores"],
                "knn_score": test_knn,
                "fraud_score": test_scores,
            }
        )
        diagnostics.to_csv(output_dir / "test_diagnostics.csv", index=False)
        summary.update({"n_test": len(test_data["ids"]), "test_predictions_path": str(test_path)})
    write_json(summary, output_dir / "summary.json")
    return summary


def _write_predictions(
    path: Path,
    ids: list[str],
    scores: np.ndarray,
    threshold: float,
    y_true: np.ndarray | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", *(["y_true"] if y_true is not None else []), "fraud_score", "label"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, (image_id, score) in enumerate(zip(ids, scores)):
            row: dict[str, object] = {
                "id": image_id,
                "fraud_score": float(score),
                "label": int(float(score) >= threshold),
            }
            if y_true is not None:
                row["y_true"] = int(y_true[index])
            writer.writerow(row)


def main() -> None:
    summary = run(parse_args())
    best = summary["best"]
    print(Path(str(summary.get("test_predictions_path", ""))).resolve() if summary.get("test_predictions_path") else "")
    print(
        f"best=k{best['k']} temp={best['temperature']} head_weight={best['head_weight']} "
        f"objective={best['selection_objective']:.6f} "
        f"audet={best['audet_proxy']:.6f} apcer={best['apcer_at_1pct_bpcer']:.6f}"
    )


if __name__ == "__main__":
    main()
