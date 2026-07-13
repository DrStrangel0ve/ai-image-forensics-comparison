from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from forensic_compare.datasets import stable_path_score  # noqa: E402
from forensic_compare.freuid import freuid_competition_path  # noqa: E402


ARCHIVE_NAME = "freuid_highres_subset.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package the deterministic FREUID high-resolution research subset for private Kaggle use."
    )
    parser.add_argument("--labels", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--holdout-type", default="EGYPT/DL")
    parser.add_argument("--max-train-samples", type=int, default=12000)
    parser.add_argument("--max-val-samples", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument(
        "--dataset-id",
        default="arnavmalani/freuid-post-freeze-highres-subset",
    )
    parser.add_argument(
        "--dataset-title",
        default="FREUID Post-freeze High-resolution Research Subset",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def limit_frame(frame: pd.DataFrame, max_samples: int, seed: int) -> pd.DataFrame:
    if max_samples <= 0 or max_samples >= len(frame):
        return frame.reset_index(drop=True)
    scored = frame.copy()
    scored["_score"] = scored["id"].astype(str).map(lambda value: stable_path_score(value, seed))
    groups = [
        group.sort_values(["_score", "id"], kind="mergesort")
        for _key, group in scored.groupby(["type", "label"], sort=True)
    ]
    selected: list[int] = []
    cursor = 0
    while len(selected) < max_samples:
        progressed = False
        for group in groups:
            if cursor < len(group):
                selected.append(int(group.index[cursor]))
                progressed = True
                if len(selected) >= max_samples:
                    break
        if not progressed:
            break
        cursor += 1
    return scored.loc[selected].drop(columns=["_score"]).reset_index(drop=True)


def select_subset(
    labels_path: Path,
    holdout_type: str,
    max_train_samples: int,
    max_val_samples: int,
    seed: int,
) -> pd.DataFrame:
    frame = pd.read_csv(labels_path)
    required = {"id", "image_path", "label", "type"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{labels_path} is missing required columns: {missing}")
    frame = frame.copy()
    frame["id"] = frame["id"].astype(str)
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    frame["type"] = frame["type"].astype(str)
    if frame["id"].duplicated().any():
        raise ValueError(f"{labels_path} contains duplicate ids")
    if holdout_type not in set(frame["type"]):
        raise ValueError(f"Unknown holdout type {holdout_type!r}")

    train = frame[frame["type"] != holdout_type].sort_values("id").reset_index(drop=True)
    val = frame[frame["type"] == holdout_type].sort_values("id").reset_index(drop=True)
    if train["label"].nunique() != 2 or val["label"].nunique() != 2:
        raise ValueError("Both selected partitions must contain both labels")
    train = limit_frame(train, max_train_samples, seed)
    val = limit_frame(val, max_val_samples, seed + 1)
    subset = pd.concat([train, val], ignore_index=True)
    subset["image_path"] = subset["image_path"].map(
        lambda value: f"train/{Path(freuid_competition_path(value)).name}"
    )
    return subset


def resolve_source_image(image_root: Path, raw_path: object) -> Path:
    raw = str(raw_path).replace("\\", "/")
    candidates = [
        Path(raw),
        image_root / raw,
        image_root / freuid_competition_path(raw),
        image_root / Path(raw).name,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Could not resolve FREUID image {raw!r}; tried {candidates}")


def selection_digest(frame: pd.DataFrame) -> str:
    payload = "\n".join(sorted(frame["id"].astype(str))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_package(
    subset: pd.DataFrame,
    source_frame: pd.DataFrame,
    image_root: Path,
    output_dir: Path,
    dataset_id: str,
    dataset_title: str,
    holdout_type: str,
    seed: int,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / ARCHIVE_NAME
    source_by_id = source_frame.assign(id=source_frame["id"].astype(str)).set_index("id")
    total_bytes = 0
    manifest = {
        "research_track": "post_freeze_highres_2026_07_13",
        "competition_eligibility": "post_freeze_research_only",
        "holdout_type": holdout_type,
        "seed": seed,
        "samples": int(len(subset)),
        "selection_sha256": selection_digest(subset),
    }

    labels_csv = subset.to_csv(index=False).encode("utf-8")
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as handle:
        handle.writestr("train_labels.csv", labels_csv)
        handle.writestr("subset_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        for row in subset.itertuples(index=False):
            source_raw = source_by_id.loc[str(row.id), "image_path"]
            source_path = resolve_source_image(image_root, source_raw)
            total_bytes += source_path.stat().st_size
            handle.write(source_path, arcname=str(row.image_path))

    manifest.update(
        {
            "archive": str(archive_path.resolve()),
            "archive_bytes": int(archive_path.stat().st_size),
            "source_image_bytes": int(total_bytes),
        }
    )
    (output_dir / "subset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "dataset-metadata.json").write_text(
        json.dumps(
            {
                "title": dataset_title,
                "id": dataset_id,
                "licenses": [{"name": "other"}],
                "isPrivate": True,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    args = parse_args()
    labels_path = Path(args.labels)
    image_root = Path(args.image_root)
    source_frame = pd.read_csv(labels_path)
    subset = select_subset(
        labels_path,
        args.holdout_type,
        args.max_train_samples,
        args.max_val_samples,
        args.seed,
    )
    summary: dict[str, object] = {
        "samples": int(len(subset)),
        "train_samples": int((subset["type"] != args.holdout_type).sum()),
        "val_samples": int((subset["type"] == args.holdout_type).sum()),
        "selection_sha256": selection_digest(subset),
        "dry_run": bool(args.dry_run),
    }
    if not args.dry_run:
        summary.update(
            write_package(
                subset,
                source_frame,
                image_root,
                Path(args.output_dir),
                args.dataset_id,
                args.dataset_title,
                args.holdout_type,
                args.seed,
            )
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
