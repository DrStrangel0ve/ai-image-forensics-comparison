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
        description="Audit an image-folder dataset for class counts, dimensions, and duplicate leakage."
    )
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-duplicate-examples", type=int, default=20)
    parser.add_argument("--phash-size", type=int, default=8)
    parser.add_argument("--near-duplicate-distance", type=int, default=4)
    parser.add_argument("--near-duplicate-dhash-distance", type=int, default=4)
    parser.add_argument(
        "--max-phash-images",
        type=int,
        default=5000,
        help="Skip near-duplicate pairwise scanning above this many images.",
    )
    parser.add_argument("--max-near-duplicate-examples", type=int, default=20)
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


def _average_hash(path: Path, size: int) -> int:
    with Image.open(path) as image:
        grayscale = image.convert("L").resize((size, size), Image.Resampling.LANCZOS)
    pixels = list(grayscale.tobytes())
    mean = sum(pixels) / len(pixels)
    value = 0
    for pixel in pixels:
        value = (value << 1) | int(pixel >= mean)
    return value


def _difference_hash(path: Path, size: int) -> int:
    with Image.open(path) as image:
        grayscale = image.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    pixels = list(grayscale.tobytes())
    value = 0
    for y in range(size):
        row = pixels[y * (size + 1) : (y + 1) * (size + 1)]
        for x in range(size):
            value = (value << 1) | int(row[x] > row[x + 1])
    return value


def _layout_folders(layout) -> list[tuple[str, Path]]:
    if layout.train and layout.test:
        return [("train", layout.train), ("test", layout.test)]
    if layout.single:
        return [("all", layout.single)]
    raise ValueError(f"Unsupported dataset layout: {layout}")


def _near_duplicate_pairs(
    rows: list[dict],
    max_average_distance: int,
    max_difference_distance: int,
    max_examples: int,
) -> tuple[list[dict], dict]:
    pairs = []
    counts = Counter()
    for left_index, left in enumerate(rows):
        left_average_hash = int(left["average_hash"], 16)
        left_difference_hash = int(left["difference_hash"], 16)
        for right in rows[left_index + 1 :]:
            if left["sha256"] == right["sha256"]:
                continue
            average_distance = (
                left_average_hash ^ int(right["average_hash"], 16)
            ).bit_count()
            if average_distance > max_average_distance:
                continue
            difference_distance = (
                left_difference_hash ^ int(right["difference_hash"], 16)
            ).bit_count()
            if difference_distance > max_difference_distance:
                continue
            cross_split = left["split"] != right["split"]
            cross_class = left["class_name"] != right["class_name"]
            counts["total"] += 1
            counts["cross_split"] += int(cross_split)
            counts["cross_class"] += int(cross_class)
            if len(pairs) < max_examples:
                pairs.append(
                    {
                        "average_distance": int(average_distance),
                        "difference_distance": int(difference_distance),
                        "cross_split": cross_split,
                        "cross_class": cross_class,
                        "left": left["path"],
                        "right": right["path"],
                        "left_split": left["split"],
                        "right_split": right["split"],
                        "left_class": left["class_name"],
                        "right_class": right["class_name"],
                    }
                )
    return sorted(
        pairs,
        key=lambda row: (
            not row["cross_split"],
            row["average_distance"],
            row["difference_distance"],
        ),
    ), counts


def audit_dataset(
    data_dir: str | Path,
    max_duplicate_examples: int = 20,
    phash_size: int = 8,
    near_duplicate_distance: int = 4,
    near_duplicate_dhash_distance: int = 4,
    max_phash_images: int = 5000,
    max_near_duplicate_examples: int = 20,
) -> dict:
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
                "average_hash": (
                    f"{_average_hash(path, phash_size):0{phash_size * phash_size // 4}x}"
                ),
                "difference_hash": (
                    f"{_difference_hash(path, phash_size):0{phash_size * phash_size // 4}x}"
                ),
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
    if len(rows) <= max_phash_images:
        near_duplicates, near_counts = _near_duplicate_pairs(
            rows,
            max_average_distance=near_duplicate_distance,
            max_difference_distance=near_duplicate_dhash_distance,
            max_examples=max_near_duplicate_examples,
        )
        near_scan_skipped = False
    else:
        near_duplicates = []
        near_counts = Counter()
        near_scan_skipped = True

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
        "phash_size": phash_size,
        "near_duplicate_distance": near_duplicate_distance,
        "near_duplicate_dhash_distance": near_duplicate_dhash_distance,
        "near_duplicate_scan_skipped": near_scan_skipped,
        "n_near_duplicate_pairs": int(near_counts["total"]),
        "n_cross_split_near_duplicate_pairs": int(near_counts["cross_split"]),
        "n_cross_class_near_duplicate_pairs": int(near_counts["cross_class"]),
        "width_min": min(widths) if widths else None,
        "width_max": max(widths) if widths else None,
        "height_min": min(heights) if heights else None,
        "height_max": max(heights) if heights else None,
        "class_counts": class_count_rows,
        "top_dimensions": top_dimensions,
        "duplicates": duplicate_summaries[:max_duplicate_examples],
        "near_duplicates": near_duplicates,
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
    near_duplicate_rows = [
        {
            "avg_dist": row["average_distance"],
            "diff_dist": row["difference_distance"],
            "cross_split": row["cross_split"],
            "cross_class": row["cross_class"],
            "left": row["left"],
            "right": row["right"],
        }
        for row in summary["near_duplicates"]
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
        f"Near-duplicate pairs: **{summary['n_near_duplicate_pairs']}**",
        f"Cross-split near-duplicate pairs: **{summary['n_cross_split_near_duplicate_pairs']}**",
        f"Cross-class near-duplicate pairs: **{summary['n_cross_class_near_duplicate_pairs']}**",
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
        "## Near-Duplicate Examples",
        "",
        "Near duplicates use average-hash Hamming distance "
        f"<= {summary['near_duplicate_distance']} over {summary['phash_size']}x"
        f"{summary['phash_size']} grayscale thumbnails, plus difference-hash Hamming distance "
        f"<= {summary['near_duplicate_dhash_distance']}.",
        "",
        "Near-duplicate scanning was skipped because the dataset exceeds the configured image cap."
        if summary["near_duplicate_scan_skipped"]
        else _markdown_table(
            near_duplicate_rows,
            ["avg_dist", "diff_dist", "cross_split", "cross_class", "left", "right"],
        )
        if near_duplicate_rows
        else "No near-duplicate pairs found.",
        "",
    ]
    (out_path / "report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    args = parse_args()
    summary = audit_dataset(
        args.data_dir,
        max_duplicate_examples=args.max_duplicate_examples,
        phash_size=args.phash_size,
        near_duplicate_distance=args.near_duplicate_distance,
        near_duplicate_dhash_distance=args.near_duplicate_dhash_distance,
        max_phash_images=args.max_phash_images,
        max_near_duplicate_examples=args.max_near_duplicate_examples,
    )
    write_audit_report(summary, args.out_dir)
    print(
        "images={n_images} duplicate_groups={n_duplicate_groups} "
        "cross_split_duplicate_groups={n_cross_split_duplicate_groups}".format(**summary)
    )
    print(f"Wrote audit to {Path(args.out_dir).resolve()}")


if __name__ == "__main__":
    main()
