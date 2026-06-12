from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forensic_compare.datasets import collect_labeled_images, discover_layout, stratified_split
from forensic_compare.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a deterministic image-folder split.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--split", choices=["train", "test", "all"], default="test")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    return parser.parse_args()


def _records_for_split(data_dir: Path, split: str, val_fraction: float, seed: int):
    layout = discover_layout(data_dir)
    if layout.single:
        records = collect_labeled_images(layout.single)
        if split == "all":
            return records, layout
        train, test = stratified_split(records, val_fraction, seed)
        return (train if split == "train" else test), layout
    if layout.train and layout.test:
        if split == "train":
            return collect_labeled_images(layout.train), layout
        if split == "test":
            return collect_labeled_images(layout.test), layout
        return collect_labeled_images(layout.train) + collect_labeled_images(layout.test), layout
    raise ValueError(f"Unsupported dataset layout: {layout}")


def _output_path(out_dir: Path, class_name: str, index: int, source_path: Path) -> Path:
    suffix = source_path.suffix.lower()
    if not suffix:
        suffix = ".jpg"
    return out_dir / class_name / f"{index:06d}{suffix}"


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"Output directory already exists and is not empty: {out_dir}")
    ensure_dir(out_dir)
    records, layout = _records_for_split(data_dir, args.split, args.val_fraction, args.seed)
    rows = []
    for index, (source_path, label, class_name) in enumerate(records):
        destination = _output_path(out_dir, class_name, index, source_path)
        ensure_dir(destination.parent)
        shutil.copy2(source_path, destination)
        rows.append(
            {
                "split": args.split,
                "class_name": class_name,
                "label": int(label),
                "source_path": str(source_path),
                "path": str(destination),
            }
        )
    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "class_name", "label", "source_path", "path"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} {args.split} images to {out_dir.resolve()}")
    print(f"Manifest: {manifest_path.resolve()}")
    print(
        "Layout: "
        f"train={layout.train if layout.train else None}, "
        f"test={layout.test if layout.test else None}, "
        f"single={layout.single if layout.single else None}"
    )


if __name__ == "__main__":
    main()
