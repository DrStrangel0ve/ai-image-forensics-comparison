from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forensic_compare.freuid_model import build_freuid_model  # noqa: E402
from forensic_compare.freuid_transforms import DocumentViewTransform  # noqa: E402
from forensic_compare.utils import resolve_device, write_json  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a frozen fine-tuned FREUID checkpoint.")
    parser.add_argument("--input-dir", default="/data")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-csv", default="/submissions/submission.csv")
    parser.add_argument("--manifest-out", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--max-images", type=int, default=0)
    return parser.parse_args()


class ImagePathDataset(Dataset):
    def __init__(self, paths: list[Path], image_size: int, grid_rows: int = 0, grid_cols: int = 0) -> None:
        self.paths = paths
        self.transform = DocumentViewTransform(
            image_size,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
        )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, path.stem


def _image_paths(input_dir: Path, recursive: bool, max_images: int) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    paths = sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if max_images > 0:
        paths = paths[:max_images]
    if not paths:
        raise FileNotFoundError(f"No image files found under {input_dir}")
    ids = [path.stem for path in paths]
    duplicates = sorted(value for value, count in Counter(ids).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate filename stems would create duplicate ids: {duplicates[:5]}")
    return paths


@torch.no_grad()
def score_checkpoint(
    checkpoint_path: Path,
    paths: list[Path],
    device: torch.device,
    batch_size: int = 64,
    num_workers: int = 4,
) -> tuple[list[str], np.ndarray, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    type_to_idx = dict(checkpoint["type_to_idx"])
    multi_view = bool(checkpoint.get("multi_view", False))
    grid_rows = int(checkpoint.get("grid_rows", 0))
    grid_cols = int(checkpoint.get("grid_cols", 0))
    forensic_residual = bool(checkpoint.get("forensic_residual", False))
    model = build_freuid_model(
        str(checkpoint["model"]),
        num_types=len(type_to_idx),
        pretrained=False,
        multi_view=multi_view,
        forensic_residual=forensic_residual,
    )
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device).eval()
    if device.type == "cuda" and not multi_view:
        model = model.to(memory_format=torch.channels_last)

    dataset = ImagePathDataset(
        paths,
        image_size=int(checkpoint["image_size"]),
        grid_rows=grid_rows,
        grid_cols=grid_cols,
    )
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(batch_size)),
        shuffle=False,
        num_workers=max(0, int(num_workers)),
        pin_memory=device.type == "cuda",
    )
    ids: list[str] = []
    scores: list[float] = []
    for images, batch_ids in tqdm(loader, desc="freuid/inference"):
        images = images.to(device, non_blocking=True)
        if device.type == "cuda" and images.ndim == 4:
            images = images.to(memory_format=torch.channels_last)
        fraud_logits, _type_logits = model(images)
        ids.extend(list(batch_ids))
        scores.extend(torch.sigmoid(fraud_logits).cpu().numpy().astype(float).tolist())

    metadata = {
        "checkpoint": str(checkpoint_path),
        "model": str(checkpoint["model"]),
        "image_size": int(checkpoint["image_size"]),
        "multi_view": multi_view,
        "forensic_residual": forensic_residual,
        "grid_rows": grid_rows,
        "grid_cols": grid_cols,
        "threshold_at_1pct_bpcer": float(checkpoint.get("threshold", 0.5)),
    }
    del model
    del checkpoint
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return ids, np.asarray(scores, dtype=float), metadata


@torch.no_grad()
def run(args: argparse.Namespace) -> dict[str, object]:
    input_dir = Path(args.input_dir)
    output_csv = Path(args.output_csv)
    manifest_out = Path(args.manifest_out) if args.manifest_out else output_csv.with_suffix(".manifest.json")
    checkpoint_path = Path(args.checkpoint)
    device = resolve_device(args.device)
    paths = _image_paths(input_dir, args.recursive, args.max_images)
    ids, score_array, checkpoint_metadata = score_checkpoint(
        checkpoint_path,
        paths,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    scores = score_array.tolist()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "label"])
        writer.writeheader()
        for image_id, score in zip(ids, scores):
            writer.writerow({"id": image_id, "label": f"{score:.10f}"})

    manifest = {
        **checkpoint_metadata,
        "input_dir": str(input_dir),
        "output_csv": str(output_csv),
        "device": str(device),
        "n_images": len(ids),
        "score_min": float(min(scores)),
        "score_max": float(max(scores)),
        "score_mean": float(sum(scores) / len(scores)),
    }
    write_json(manifest, manifest_out)
    return manifest


def main() -> None:
    manifest = run(parse_args())
    print(Path(str(manifest["output_csv"])).resolve())
    print(f"n_images={manifest['n_images']} score_mean={manifest['score_mean']:.6f}")


if __name__ == "__main__":
    main()
