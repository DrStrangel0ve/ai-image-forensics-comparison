from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forensic_compare.freuid_model import build_freuid_model
from forensic_compare.utils import resolve_device, write_json


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


class Letterbox:
    def __init__(self, size: int) -> None:
        self.size = int(size)

    def __call__(self, image: Image.Image) -> Image.Image:
        return ImageOps.pad(
            image,
            (self.size, self.size),
            method=Image.Resampling.BICUBIC,
            color=(127, 127, 127),
            centering=(0.5, 0.5),
        )


class ImagePathDataset(Dataset):
    def __init__(self, paths: list[Path], image_size: int) -> None:
        self.paths = paths
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
def run(args: argparse.Namespace) -> dict[str, object]:
    input_dir = Path(args.input_dir)
    output_csv = Path(args.output_csv)
    manifest_out = Path(args.manifest_out) if args.manifest_out else output_csv.with_suffix(".manifest.json")
    checkpoint_path = Path(args.checkpoint)
    device = resolve_device(args.device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    type_to_idx = dict(checkpoint["type_to_idx"])
    model = build_freuid_model(
        str(checkpoint["model"]),
        num_types=len(type_to_idx),
        pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device).eval()
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    paths = _image_paths(input_dir, args.recursive, args.max_images)
    dataset = ImagePathDataset(paths, image_size=int(checkpoint["image_size"]))
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(args.batch_size)),
        shuffle=False,
        num_workers=max(0, int(args.num_workers)),
        pin_memory=device.type == "cuda",
    )
    ids: list[str] = []
    scores: list[float] = []
    for images, batch_ids in tqdm(loader, desc="freuid/inference"):
        images = images.to(device, non_blocking=True)
        if device.type == "cuda":
            images = images.to(memory_format=torch.channels_last)
        fraud_logits, _type_logits = model(images)
        ids.extend(list(batch_ids))
        scores.extend(torch.sigmoid(fraud_logits).cpu().numpy().astype(float).tolist())

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "label"])
        writer.writeheader()
        for image_id, score in zip(ids, scores):
            writer.writerow({"id": image_id, "label": f"{score:.10f}"})

    manifest = {
        "checkpoint": str(checkpoint_path),
        "model": str(checkpoint["model"]),
        "image_size": int(checkpoint["image_size"]),
        "input_dir": str(input_dir),
        "output_csv": str(output_csv),
        "device": str(device),
        "n_images": len(ids),
        "score_min": float(min(scores)),
        "score_max": float(max(scores)),
        "score_mean": float(sum(scores) / len(scores)),
        "threshold_at_1pct_bpcer": float(checkpoint.get("threshold", 0.5)),
    }
    write_json(manifest, manifest_out)
    return manifest


def main() -> None:
    manifest = run(parse_args())
    print(Path(str(manifest["output_csv"])).resolve())
    print(f"n_images={manifest['n_images']} score_mean={manifest['score_mean']:.6f}")


if __name__ == "__main__":
    main()
