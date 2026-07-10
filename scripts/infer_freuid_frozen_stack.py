from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forensic_compare.conventional import extract_feature_set
from forensic_compare.foundation import build_frozen_encoder, encode_batch, frozen_encoder_transform
from forensic_compare.utils import resolve_device, write_json


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen FREUID submission stack on a flat image directory and "
            "write Kaggle id,label fraud-score predictions."
        )
    )
    parser.add_argument("--input-dir", default="/data", help="Directory containing FREUID test images.")
    parser.add_argument("--output-csv", default="/submissions/submission.csv")
    parser.add_argument("--combined-v4-model", required=True, help="Path to combined_v4 HGB classifier.joblib.")
    parser.add_argument("--convnext-model", required=True, help="Path to ConvNeXt-Tiny logistic classifier.joblib.")
    parser.add_argument("--manifest-out", default=None)
    parser.add_argument("--torch-home", default=None, help="Runtime TORCH_HOME containing hub/checkpoints/*.pth.")
    parser.add_argument("--combined-v4-weight", type=float, default=0.7)
    parser.add_argument("--convnext-weight", type=float, default=0.3)
    parser.add_argument("--combined-image-size", type=int, default=128)
    parser.add_argument("--encoder", default="convnext_tiny")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--recursive", action="store_true", help="Search image files recursively.")
    parser.add_argument("--max-images", type=int, default=0, help="Smoke-test limit; 0 means all images.")
    return parser.parse_args()


def _image_paths(input_dir: Path, recursive: bool, max_images: int) -> list[Path]:
    iterator = input_dir.rglob("*") if recursive else input_dir.iterdir()
    paths = sorted(path for path in iterator if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if max_images > 0:
        paths = paths[:max_images]
    if not paths:
        raise FileNotFoundError(f"No image files found under {input_dir}")
    ids = [path.stem for path in paths]
    duplicate_ids = sorted(value for value, count in Counter(ids).items() if count > 1)
    if duplicate_ids:
        raise ValueError(f"Duplicate filename stems would create duplicate Kaggle ids: {duplicate_ids[:5]}")
    return paths


def _classifier_scores(model, features: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(features)[:, 1], dtype=np.float64)
    raw = np.asarray(model.decision_function(features), dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-raw))


def _load_rgb_tensor(path: Path, transform) -> torch.Tensor:
    with Image.open(path) as image:
        return transform(image.convert("RGB"))


def _combined_v4_scores(paths: list[Path], model_path: Path, image_size: int) -> np.ndarray:
    model = joblib.load(model_path)
    features = []
    for path in tqdm(paths, desc="combined_v4/features"):
        features.append(extract_feature_set(path, image_size=image_size, feature_set="combined_v4"))
    return _classifier_scores(model, np.vstack(features).astype(np.float32))


@torch.no_grad()
def _convnext_scores(
    paths: list[Path],
    model_path: Path,
    encoder_name: str,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    classifier = joblib.load(model_path)
    spec = build_frozen_encoder(encoder_name, pretrained=True)
    transform = frozen_encoder_transform(spec.image_size, spec.mean, spec.std)
    encoder = spec.model.to(device)
    embeddings: list[np.ndarray] = []
    batch: list[torch.Tensor] = []

    def flush() -> None:
        if not batch:
            return
        tensor = torch.stack(batch).to(device)
        encoded = encode_batch(encoder, tensor).detach().cpu().numpy().astype(np.float32)
        embeddings.append(encoded)
        batch.clear()

    for path in tqdm(paths, desc=f"{encoder_name}/embeddings"):
        batch.append(_load_rgb_tensor(path, transform))
        if len(batch) >= batch_size:
            flush()
    flush()
    return _classifier_scores(classifier, np.vstack(embeddings))


def run(args: argparse.Namespace) -> dict[str, object]:
    if args.torch_home:
        os.environ["TORCH_HOME"] = str(Path(args.torch_home))

    input_dir = Path(args.input_dir)
    output_csv = Path(args.output_csv)
    manifest_out = Path(args.manifest_out) if args.manifest_out else output_csv.with_suffix(".manifest.json")
    paths = _image_paths(input_dir, recursive=args.recursive, max_images=args.max_images)
    device = resolve_device(args.device)

    combined_scores = _combined_v4_scores(paths, Path(args.combined_v4_model), args.combined_image_size)
    convnext_scores = _convnext_scores(
        paths=paths,
        model_path=Path(args.convnext_model),
        encoder_name=args.encoder,
        batch_size=max(1, int(args.batch_size)),
        device=device,
    )

    total_weight = float(args.combined_v4_weight + args.convnext_weight)
    if total_weight <= 0:
        raise ValueError("Fusion weights must sum to a positive value")
    fused = (args.combined_v4_weight * combined_scores + args.convnext_weight * convnext_scores) / total_weight
    fused = np.clip(fused, 0.0, 1.0)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "label"])
        writer.writeheader()
        for path, score in zip(paths, fused):
            writer.writerow({"id": path.stem, "label": f"{float(score):.10f}"})

    manifest = {
        "input_dir": str(input_dir),
        "output_csv": str(output_csv),
        "n_images": int(len(paths)),
        "device": str(device),
        "encoder": args.encoder,
        "combined_v4_model": str(args.combined_v4_model),
        "convnext_model": str(args.convnext_model),
        "combined_v4_weight": float(args.combined_v4_weight),
        "convnext_weight": float(args.convnext_weight),
        "score_min": float(fused.min()),
        "score_max": float(fused.max()),
        "score_mean": float(fused.mean()),
    }
    write_json(manifest, manifest_out)
    return manifest


def main() -> None:
    manifest = run(parse_args())
    print(Path(str(manifest["output_csv"])).resolve())
    print(f"n_images={manifest['n_images']} score_mean={manifest['score_mean']:.6f}")


if __name__ == "__main__":
    main()
