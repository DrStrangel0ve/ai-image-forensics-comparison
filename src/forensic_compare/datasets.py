from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Iterable

IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
FAKE_TOKENS = {
    "ai",
    "artificial",
    "diffusion",
    "fake",
    "gan",
    "generated",
    "synthetic",
}
REAL_TOKENS = {"authentic", "natural", "photo", "photograph", "real"}


@dataclass(frozen=True)
class DataLayout:
    """Resolved image-folder layout.

    If ``train`` and ``test`` are present, scripts use those fixed splits.
    If only ``single`` is present, scripts create a stratified train/test split.
    """

    train: Path | None
    test: Path | None
    single: Path | None = None


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def image_files(root: Path) -> list[Path]:
    return [path for path in root.rglob("*") if is_image(path)]


def class_kind(class_name: str) -> str | None:
    normalized = class_name.lower().replace("-", "_").replace(" ", "_")
    tokens = set(normalized.split("_")) | {normalized}
    if tokens & FAKE_TOKENS:
        return "fake"
    if tokens & REAL_TOKENS:
        return "real"
    return None


def class_dirs(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    dirs = []
    for child in sorted(folder.iterdir()):
        if child.is_dir() and any(is_image(path) for path in child.rglob("*")):
            dirs.append(child)
    return dirs


def has_binary_classes(folder: Path) -> bool:
    kinds = {class_kind(path.name) for path in class_dirs(folder)}
    return "fake" in kinds and "real" in kinds


def _split_score(path: Path, wanted: Iterable[str]) -> int:
    name = path.name.lower()
    wanted_set = set(wanted)
    if name in wanted_set:
        return 3
    if any(token in name for token in wanted_set):
        return 2
    return 0


def discover_layout(root: str | Path, max_depth: int = 4) -> DataLayout:
    """Find train/test class folders under a Kaggle-style image dataset root."""

    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root_path}")

    candidates = [root_path]
    candidates.extend(path for path in root_path.rglob("*") if path.is_dir())
    candidates = [
        path
        for path in candidates
        if len(path.relative_to(root_path).parts) <= max_depth and has_binary_classes(path)
    ]
    if not candidates:
        raise ValueError(
            f"No image-folder layout with real/fake class directories was found under {root_path}"
        )

    train_names = ("train", "training")
    test_names = ("test", "testing", "val", "valid", "validation")
    train_candidates = [(path, _split_score(path, train_names)) for path in candidates]
    test_candidates = [(path, _split_score(path, test_names)) for path in candidates]
    train_candidates = [item for item in train_candidates if item[1] > 0]
    test_candidates = [item for item in test_candidates if item[1] > 0]

    if train_candidates and test_candidates:
        train_path = max(train_candidates, key=lambda item: (item[1], -len(item[0].parts)))[0]
        test_path = max(test_candidates, key=lambda item: (item[1], -len(item[0].parts)))[0]
        if train_path != test_path:
            return DataLayout(train=train_path, test=test_path)

    # Fall back to a single folder with class directories.
    single = min(candidates, key=lambda path: len(path.parts))
    return DataLayout(train=None, test=None, single=single)


def collect_labeled_images(folder: str | Path) -> list[tuple[Path, int, str]]:
    """Return ``(path, y, class_name)`` records where y=1 means generated/fake."""

    folder_path = Path(folder)
    records: list[tuple[Path, int, str]] = []
    for class_dir in class_dirs(folder_path):
        kind = class_kind(class_dir.name)
        if kind is None:
            continue
        label = 1 if kind == "fake" else 0
        records.extend((path, label, class_dir.name) for path in image_files(class_dir))
    if not records:
        raise ValueError(f"No labeled real/fake images found in {folder_path}")
    return sorted(records, key=lambda item: str(item[0]))


def stratified_split(
    records: list[tuple[Path, int, str]],
    test_fraction: float,
    seed: int,
) -> tuple[list[tuple[Path, int, str]], list[tuple[Path, int, str]]]:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    rng = Random(seed)
    by_label: dict[int, list[tuple[Path, int, str]]] = {}
    for record in records:
        by_label.setdefault(record[1], []).append(record)

    train: list[tuple[Path, int, str]] = []
    test: list[tuple[Path, int, str]] = []
    for label_records in by_label.values():
        shuffled = label_records[:]
        rng.shuffle(shuffled)
        n_test = max(1, int(round(len(shuffled) * test_fraction)))
        test.extend(shuffled[:n_test])
        train.extend(shuffled[n_test:])
    return train, test


def limit_records(
    records: list[tuple[Path, int, str]],
    max_samples: int | None,
    seed: int,
) -> list[tuple[Path, int, str]]:
    if max_samples is None or max_samples <= 0 or max_samples >= len(records):
        return records
    rng = Random(seed)
    by_label: dict[int, list[tuple[Path, int, str]]] = {}
    for record in records:
        by_label.setdefault(record[1], []).append(record)

    per_label = max(1, max_samples // max(1, len(by_label)))
    selected: list[tuple[Path, int, str]] = []
    for label_records in by_label.values():
        shuffled = label_records[:]
        rng.shuffle(shuffled)
        selected.extend(shuffled[:per_label])

    if len(selected) < max_samples:
        remaining = [record for record in records if record not in selected]
        rng.shuffle(remaining)
        selected.extend(remaining[: max_samples - len(selected)])
    return selected[:max_samples]
