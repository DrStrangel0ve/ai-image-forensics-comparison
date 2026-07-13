from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forensic_compare.datasets import stable_path_score
from forensic_compare.freuid import freuid_metrics
from forensic_compare.freuid_model import build_freuid_model
from forensic_compare.freuid_transforms import DocumentViewTransform
from forensic_compare.metrics import binary_metrics
from forensic_compare.transforms import ROBUSTNESS_VARIANTS, apply_robustness_variant
from forensic_compare.utils import ensure_dir, resolve_device, write_json


VARIANTS = ("clean", *ROBUSTNESS_VARIANTS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a FREUID checkpoint under deterministic capture stresses.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--max-samples", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=79)
    return parser.parse_args()


def _limited_frame(path: Path, max_samples: int, seed: int) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"id", "label"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Data CSV is missing required columns: {missing}")
    frame = frame.copy()
    frame["id"] = frame["id"].astype(str)
    if max_samples <= 0 or max_samples >= len(frame):
        return frame.reset_index(drop=True)
    frame["_score"] = frame["id"].map(lambda value: stable_path_score(value, seed))
    groups = [column for column in ("type", "label") if column in frame.columns]
    selected = []
    per_group = max(1, max_samples // max(1, frame.groupby(groups).ngroups))
    for _key, group in frame.groupby(groups, sort=True):
        selected.append(group.sort_values(["_score", "id"]).head(per_group))
    limited = pd.concat(selected, ignore_index=True)
    if len(limited) < max_samples:
        remainder = frame[~frame["id"].isin(limited["id"])].sort_values(["_score", "id"])
        limited = pd.concat([limited, remainder.head(max_samples - len(limited))], ignore_index=True)
    return limited.head(max_samples).drop(columns=["_score"]).reset_index(drop=True)


class RobustnessDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform: DocumentViewTransform, variant: str) -> None:
        self.rows = frame.to_dict("records")
        self.transform = transform
        self.variant = variant

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        raw_path = row.get("local_path", row.get("image_path"))
        if raw_path is None:
            raise ValueError("Each data row must contain local_path or image_path")
        with Image.open(str(raw_path)) as image:
            image = image.convert("RGB")
            if self.variant != "clean":
                image = apply_robustness_variant(image, self.variant)
            tensor = self.transform(image)
        return tensor, int(row["label"]), str(row["id"])


@torch.no_grad()
def _score_variant(model, loader: DataLoader, device: torch.device, variant: str) -> tuple[list[dict[str, object]], dict[str, object]]:
    model.eval()
    rows: list[dict[str, object]] = []
    for images, labels, ids in tqdm(loader, desc=f"robustness/{variant}", leave=False):
        images = images.to(device, non_blocking=True)
        if device.type == "cuda" and images.ndim == 4:
            images = images.to(memory_format=torch.channels_last)
        fraud_logits, _type_logits = model(images)
        scores = torch.sigmoid(fraud_logits).cpu().numpy().astype(float)
        rows.extend(
            {"id": image_id, "y_true": int(label), "fraud_score": float(score), "variant": variant}
            for image_id, label, score in zip(ids, labels.numpy(), scores)
        )
    y_true = [int(row["y_true"]) for row in rows]
    scores = [float(row["fraud_score"]) for row in rows]
    metrics = binary_metrics(y_true, scores)
    metrics.update(freuid_metrics(y_true, scores))
    metrics["variant"] = variant
    return rows, metrics


def run(args: argparse.Namespace) -> dict[str, object]:
    output_dir = ensure_dir(args.output_dir)
    device = resolve_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    type_to_idx = dict(checkpoint["type_to_idx"])
    multi_view = bool(checkpoint.get("multi_view", False))
    forensic_residual = bool(checkpoint.get("forensic_residual", False))
    model = build_freuid_model(
        str(checkpoint["model"]),
        num_types=len(type_to_idx),
        pretrained=False,
        multi_view=multi_view,
        forensic_residual=forensic_residual,
        view_pooling=str(checkpoint.get("view_pooling", "attention")),
        freeze_encoder=bool(checkpoint.get("freeze_encoder", False)),
        lora_rank=int(checkpoint.get("lora_rank", 0)),
        lora_alpha=float(checkpoint.get("lora_alpha", 16.0)),
        view_chunk_size=int(checkpoint.get("view_chunk_size", 0)),
    )
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device).eval()
    if device.type == "cuda" and not multi_view:
        model = model.to(memory_format=torch.channels_last)

    transform = DocumentViewTransform(
        int(checkpoint["image_size"]),
        grid_rows=int(checkpoint.get("grid_rows", 0)),
        grid_cols=int(checkpoint.get("grid_cols", 0)),
        view_mode=str(checkpoint.get("view_mode", "auto")),
        five_crop_zoom=float(checkpoint.get("five_crop_zoom", 1.15)),
    )
    frame = _limited_frame(Path(args.data_csv), args.max_samples, args.seed)
    metric_rows: list[dict[str, object]] = []
    for variant in args.variants:
        dataset = RobustnessDataset(frame, transform, variant)
        loader = DataLoader(
            dataset,
            batch_size=max(1, int(args.batch_size)),
            shuffle=False,
            num_workers=max(0, int(args.num_workers)),
            pin_memory=device.type == "cuda",
        )
        prediction_rows, metrics = _score_variant(model, loader, device, variant)
        metric_rows.append(metrics)
        with (output_dir / f"predictions_{variant}.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "y_true", "fraud_score", "variant"])
            writer.writeheader()
            writer.writerows(prediction_rows)

    summary_frame = pd.DataFrame(metric_rows)
    summary_frame.to_csv(output_dir / "summary.csv", index=False)
    summary = {
        "checkpoint": str(args.checkpoint),
        "data_csv": str(args.data_csv),
        "device": str(device),
        "n_samples": int(len(frame)),
        "variants": list(args.variants),
        "rows": metric_rows,
    }
    write_json(summary, output_dir / "summary.json")
    return summary


def main() -> None:
    args = parse_args()
    summary = run(args)
    print(Path(args.output_dir).resolve())
    for row in summary["rows"]:
        print(
            f"{row['variant']}: audet={row['audet_proxy']:.6f} "
            f"apcer={row['apcer_at_1pct_bpcer']:.6f} auc={row['roc_auc']:.6f}"
        )


if __name__ == "__main__":
    main()
