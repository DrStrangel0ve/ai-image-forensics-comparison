from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forensic_compare.catalog import load_catalog
from forensic_compare.utils import ensure_dir


DEFAULT_REAL_LABELS = ("0", "real", "Real")
DEFAULT_FAKE_LABELS = ("1", "fake", "AI-Generated")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a Hugging Face image dataset to real/fake image-folder layout."
    )
    parser.add_argument("--dataset-key", default=None, help="Key from configs/datasets.json.")
    parser.add_argument("--repo-id", default=None, help="Hugging Face dataset repo. Overrides --dataset-key.")
    parser.add_argument("--config", default="default")
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--image-column", default="Image")
    parser.add_argument("--label-column", default="Label_A")
    parser.add_argument("--caption-column", default="Caption")
    parser.add_argument("--source-label-column", default="Label_B")
    parser.add_argument("--real-label", action="append", default=None)
    parser.add_argument("--fake-label", action="append", default=None)
    parser.add_argument("--max-per-class-per-split", type=int, default=0)
    parser.add_argument(
        "--max-real-per-split",
        type=int,
        default=0,
        help="Optional real-image cap per split. Defaults to --max-per-class-per-split when unset.",
    )
    parser.add_argument(
        "--max-per-source-per-split",
        type=int,
        default=0,
        help="Optional cap for each generated source label, such as each Defactify generator.",
    )
    parser.add_argument(
        "--fake-source-label",
        action="append",
        default=[],
        help="Generated source label to keep when using --max-per-source-per-split. Repeat as needed.",
    )
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--shuffle-buffer", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--image-format", choices=["jpg", "png"], default="jpg")
    return parser.parse_args()


def _repo_id(args: argparse.Namespace) -> str:
    if args.repo_id:
        return args.repo_id
    if not args.dataset_key:
        raise SystemExit("Provide --dataset-key or --repo-id")
    entry = load_catalog()[args.dataset_key]
    if entry.source != "huggingface":
        raise SystemExit(f"{entry.key} is {entry.source}, not a Hugging Face dataset")
    return entry.ref


def _label_kind(value: Any, real_labels: set[str], fake_labels: set[str]) -> str:
    normalized = str(value)
    if normalized in real_labels:
        return "real"
    if normalized in fake_labels:
        return "ai_generated"
    raise ValueError(f"Unsupported label value {value!r}; update --real-label/--fake-label")


def _label_sets(real_values: list[str] | None, fake_values: list[str] | None) -> tuple[set[str], set[str]]:
    real_labels = set(real_values or DEFAULT_REAL_LABELS)
    fake_labels = set(fake_values or DEFAULT_FAKE_LABELS)
    overlap = real_labels & fake_labels
    if overlap:
        raise ValueError(f"Labels cannot be both real and fake: {sorted(overlap)}")
    return real_labels, fake_labels


def _coerce_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return ImageOps.exif_transpose(value).convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            return Image.open(value["path"]).convert("RGB")
    if isinstance(value, (str, Path)):
        return Image.open(value).convert("RGB")
    raise TypeError(f"Unsupported image value type: {type(value).__name__}")


def _load_split(repo_id: str, config: str | None, split: str, streaming: bool):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install dataset support first: python -m pip install datasets") from exc
    kwargs = {"split": split, "streaming": streaming}
    if config:
        return load_dataset(repo_id, config, **kwargs)
    return load_dataset(repo_id, **kwargs)


def _save_image(image: Image.Image, path: Path, image_format: str) -> None:
    ensure_dir(path.parent)
    if image_format == "jpg":
        image.save(path, format="JPEG", quality=95)
    else:
        image.save(path, format="PNG")


def _cap(value: int) -> int | None:
    return value if value > 0 else None


def _split_done(
    counts: Counter,
    source_counts: Counter,
    real_cap: int | None,
    fake_cap: int | None,
    source_cap: int | None,
    fake_source_labels: set[str],
) -> bool:
    real_done = real_cap is None or counts["real"] >= real_cap
    if source_cap is not None:
        if fake_source_labels:
            fake_done = all(source_counts[label] >= source_cap for label in fake_source_labels)
        elif fake_cap is not None:
            fake_done = counts["ai_generated"] >= fake_cap
        else:
            fake_done = False
    else:
        fake_done = fake_cap is None or counts["ai_generated"] >= fake_cap
    return real_done and fake_done


def _should_skip(
    kind: str,
    source_label: str,
    counts: Counter,
    source_counts: Counter,
    real_cap: int | None,
    fake_cap: int | None,
    source_cap: int | None,
    fake_source_labels: set[str],
) -> bool:
    if kind == "real":
        return real_cap is not None and counts["real"] >= real_cap
    if source_cap is not None:
        if fake_source_labels and source_label not in fake_source_labels:
            return True
        if source_counts[source_label] >= source_cap:
            return True
    return fake_cap is not None and counts["ai_generated"] >= fake_cap


def export_split(args: argparse.Namespace, repo_id: str, split: str, writer: csv.DictWriter) -> Counter:
    dataset = _load_split(repo_id, args.config, split, args.streaming)
    if args.shuffle_buffer > 0 and hasattr(dataset, "shuffle"):
        dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    counts: Counter = Counter()
    source_counts: Counter = Counter()
    real_labels, fake_labels = _label_sets(args.real_label, args.fake_label)
    max_per_class = _cap(args.max_per_class_per_split)
    real_cap = _cap(args.max_real_per_split) or max_per_class
    fake_cap = max_per_class
    source_cap = _cap(args.max_per_source_per_split)
    fake_source_labels = set(args.fake_source_label)
    progress = tqdm(dataset, desc=f"export/{split}")
    for index, row in enumerate(progress):
        kind = _label_kind(row[args.label_column], real_labels, fake_labels)
        source_label = str(row.get(args.source_label_column, ""))
        if _should_skip(
            kind,
            source_label,
            counts,
            source_counts,
            real_cap,
            fake_cap,
            source_cap,
            fake_source_labels,
        ):
            if _split_done(counts, source_counts, real_cap, fake_cap, source_cap, fake_source_labels):
                break
            continue
        image = _coerce_image(row[args.image_column])
        suffix = "jpg" if args.image_format == "jpg" else "png"
        out_path = (
            Path(args.out_dir)
            / split
            / kind
            / f"{split}_{kind}_{counts[kind]:06d}.{suffix}"
        )
        _save_image(image, out_path, args.image_format)
        writer.writerow(
            {
                "split": split,
                "index": index,
                "class_name": kind,
                "label": 0 if kind == "real" else 1,
                "source_label": source_label,
                "caption": row.get(args.caption_column, ""),
                "path": str(out_path),
            }
        )
        counts[kind] += 1
        if kind == "ai_generated":
            source_counts[source_label] += 1
    return counts


def main() -> None:
    args = parse_args()
    repo_id = _repo_id(args)
    out_dir = ensure_dir(args.out_dir)
    metadata_path = out_dir / "metadata.csv"
    with metadata_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["split", "index", "class_name", "label", "source_label", "caption", "path"],
        )
        writer.writeheader()
        totals: dict[str, dict[str, int]] = {}
        for split in args.splits:
            counts = export_split(args, repo_id, split, writer)
            totals[split] = {key: int(value) for key, value in counts.items()}
    print(f"Exported {repo_id} to {out_dir.resolve()}")
    for split, counts in totals.items():
        print(f"{split}: {counts}")


if __name__ == "__main__":
    main()
