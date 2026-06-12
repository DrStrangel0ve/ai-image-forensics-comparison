from __future__ import annotations

import argparse
import hashlib
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image

from forensic_compare.datasets import collect_labeled_images, discover_layout
from forensic_compare.utils import ensure_dir, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit an image-folder dataset for class counts, dimensions, and exact duplicates."
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-duplicate-examples", type=int, default=20)
    return parser.parse_args()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _layout_folders(layout) -> list[tuple[str, Path]]:
    if layout.train and layout.test:
        return [("train", layout.train), ("test", layout.test)]
    if layout.single:
        return [("all", layout.single)]
    raise ValueError(f"Unsupported dataset layout: {layout}")


def audit_dataset(data_dir: str | Path, max_duplicate_examples: int = 20) -> dict:
    root = Path(data_dir).expanduser().resolve()
    layout = discover_layout(root)
    rows = []
    duplicate_groups: dict[str, list[dict]] = defaultdict(list)
    class_counts = Counter()
    dimension_counts = Counter()
    widths = []
    heights = []

    for split, folder in _layout_folders(layout):
        for path, label, class_name in collect_labeled_images(folder):
            digest = _sha256_file(path)
            width, height = _image_size(path)
            rel_path = path.relative_to(root).as_posix()
            row = {
                "path": rel_path,
                "split": split,
                "class_name": class_name,
                "label": int(label),
                "sha256": digest,
                "width": int(width),
                "height": int(height),
                "bytes": int(path.stat().st_size),
            }
            rows.append(row)
            duplicate_groups[digest].append(row)
            class_counts[(split, class_name, int(label))] += 1
            dimension_counts[(int(width), int(height))] += 1
            widths.append(int(width))
            heights.append(int(height))

    duplicates = [group for group in duplicate_groups.values() if len(group) > 1]
    duplicate_summaries = []
    for group in duplicates:
        splits = sorted({row["split"] for row in group})
        classes = sorted({row["class_name"] for row in group})
        duplicate_summaries.append(
            {
                "sha256": group[0]["sha256"],
                "n_images": len(group),
                "splits": splits,
                "classes": classes,
                "cross_split": len(splits) > 1,
                "cross_class": len(classes) > 1,
                "paths": [row["path"] for row in group],
            }
        )
    duplicate_summaries = sorted(
        duplicate_summaries,
        key=lambda row: (not row["cross_split"], not row["cross_class"], row["sha256"]),
    )

    class_count_rows = [
        {"split": split, "class_name": class_name, "label": label, "n_images": count}
        for (split, class_name, label), count in sorted(class_counts.items())
    ]
    top_dimensions = [
        {"width": width, "height": height, "n_images": count}
        for (width, height), count in dimension_counts.most_common(10)
    ]
    summary = {
        "data_dir": str(root),
        "layout": {
            "train": str(layout.train) if layout.train else None,
            "test": str(layout.test) if layout.test else None,
            "single": str(layout.single) if layout.single else None,
        },
        "n_images": len(rows),
        "n_unique_sha256": len(duplicate_groups),
        "n_duplicate_groups": len(duplicates),
        "n_duplicate_images": sum(len(group) for group in duplicates),
        "n_cross_split_duplicate_groups": sum(
            1 for group in duplicate_summaries if group["cross_split"]
        ),
        "n_cross_class_duplicate_groups": sum(
            1 for group in duplicate_summaries if group["cross_class"]
        ),
        "width_min": min(widths) if widths else None,
        "width_max": max(widths) if widths else None,
        "height_min": min(heights) if heights else None,
        "height_max": max(heights) if heights else None,
        "class_counts": class_count_rows,
        "top_dimensions": top_dimensions,
        "duplicates": duplicate_summaries[:max_duplicate_examples],
    }
    return summary


def _markdown_table(rows: list[dict], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def write_audit_report(summary: dict, out_dir: str | Path) -> None:
    out_path = ensure_dir(out_dir)
    write_json(summary, out_path / "audit.json")
    duplicate_rows = [
        {
            "n_images": row["n_images"],
            "cross_split": row["cross_split"],
            "cross_class": row["cross_class"],
            "classes": ",".join(row["classes"]),
            "example": row["paths"][0],
        }
        for row in summary["duplicates"]
    ]
    report = [
        "# Image Dataset Audit",
        "",
        f"Dataset: `{summary['data_dir']}`",
        "",
        f"Images: **{summary['n_images']}**",
        f"Unique SHA-256 hashes: **{summary['n_unique_sha256']}**",
        f"Duplicate groups: **{summary['n_duplicate_groups']}**",
        f"Cross-split duplicate groups: **{summary['n_cross_split_duplicate_groups']}**",
        f"Cross-class duplicate groups: **{summary['n_cross_class_duplicate_groups']}**",
        f"Width range: **{summary['width_min']}..{summary['width_max']}**",
        f"Height range: **{summary['height_min']}..{summary['height_max']}**",
        "",
        "## Class Counts",
        "",
        _markdown_table(summary["class_counts"], ["split", "class_name", "label", "n_images"]),
        "",
        "## Common Dimensions",
        "",
        _markdown_table(summary["top_dimensions"], ["width", "height", "n_images"]),
        "",
        "## Duplicate Examples",
        "",
        _markdown_table(
            duplicate_rows,
            ["n_images", "cross_split", "cross_class", "classes", "example"],
        )
        if duplicate_rows
        else "No exact duplicate groups found.",
        "",
    ]
    (out_path / "report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary = audit_dataset(args.data_dir, max_duplicate_examples=args.max_duplicate_examples)
    write_audit_report(summary, args.out_dir)
    print(
        "images={n_images} duplicate_groups={n_duplicate_groups} "
        "cross_split_duplicate_groups={n_cross_split_duplicate_groups}".format(**summary)
    )
    print(f"Wrote audit to {Path(args.out_dir).resolve()}")


if __name__ == "__main__":
    main()
