from __future__ import annotations

import argparse
import csv
import sys
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageFilter
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forensic_compare.datasets import collect_labeled_images, discover_layout
from forensic_compare.utils import ensure_dir


VARIANTS = ("jpeg70", "blur1", "resize_half", "crop85")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create transformed image-folder variants for robustness testing."
    )
    parser.add_argument("--data-dir", required=True, help="Source image-folder dataset.")
    parser.add_argument("--out-dir", required=True, help="Output folder for variant datasets.")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument(
        "--format",
        choices=["jpg", "png"],
        default="jpg",
        help="Output image format. JPEG keeps robustness tests close to web/social sharing.",
    )
    return parser.parse_args()


def _split_folders(data_dir: Path) -> list[tuple[str, Path]]:
    layout = discover_layout(data_dir)
    folders = []
    if layout.train:
        folders.append((layout.train.name, layout.train))
    if layout.test:
        folders.append((layout.test.name, layout.test))
    if layout.single:
        folders.append((layout.single.name, layout.single))
    return folders


def _jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as jpeg:
        return jpeg.convert("RGB")


def _center_crop_resize(image: Image.Image, fraction: float) -> Image.Image:
    width, height = image.size
    crop_width = max(1, int(round(width * fraction)))
    crop_height = max(1, int(round(height * fraction)))
    left = (width - crop_width) // 2
    top = (height - crop_height) // 2
    cropped = image.crop((left, top, left + crop_width, top + crop_height))
    return cropped.resize((width, height), Image.Resampling.BICUBIC)


def _resize_half_roundtrip(image: Image.Image) -> Image.Image:
    width, height = image.size
    down = image.resize((max(1, width // 2), max(1, height // 2)), Image.Resampling.BICUBIC)
    return down.resize((width, height), Image.Resampling.BICUBIC)


def _apply_variant(image: Image.Image, variant: str) -> Image.Image:
    image = image.convert("RGB")
    if variant == "jpeg70":
        return _jpeg_roundtrip(image, quality=70)
    if variant == "blur1":
        return image.filter(ImageFilter.GaussianBlur(radius=1.0))
    if variant == "resize_half":
        return _resize_half_roundtrip(image)
    if variant == "crop85":
        return _center_crop_resize(image, fraction=0.85)
    raise ValueError(f"Unsupported variant: {variant}")


def _save_image(image: Image.Image, path: Path, output_format: str) -> None:
    ensure_dir(path.parent)
    if output_format == "jpg":
        image.save(path, format="JPEG", quality=95)
    else:
        image.save(path, format="PNG")


def _output_path(
    out_dir: Path,
    variant: str,
    split: str,
    class_name: str,
    index: int,
    output_format: str,
) -> Path:
    suffix = ".jpg" if output_format == "jpg" else ".png"
    return out_dir / variant / split / class_name / f"{index:06d}{suffix}"


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = ensure_dir(args.out_dir)
    manifest_path = out_dir / "manifest.csv"
    rows = []
    for split, folder in _split_folders(data_dir):
        records = collect_labeled_images(folder)
        for variant in args.variants:
            for index, (path, label, class_name) in enumerate(
                tqdm(records, desc=f"{variant}/{split}")
            ):
                with Image.open(path) as image:
                    transformed = _apply_variant(image, variant)
                out_path = _output_path(out_dir, variant, split, class_name, index, args.format)
                _save_image(transformed, out_path, args.format)
                rows.append(
                    {
                        "variant": variant,
                        "split": split,
                        "class_name": class_name,
                        "label": int(label),
                        "source_path": str(path),
                        "path": str(out_path),
                    }
                )
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["variant", "split", "class_name", "label", "source_path", "path"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote robustness variants to {out_dir.resolve()}")
    print(f"Manifest: {manifest_path.resolve()}")


if __name__ == "__main__":
    main()
