from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd
import torch


REPO_URL = "https://github.com/DrStrangel0ve/ai-image-forensics-comparison.git"
REPO_REF = "post-freeze-highres-kaggle-v4"
COMPETITION_SLUG = "the-freuid-challenge-2026-ijcai-ecai"
HOLDOUT_TYPE = "EGYPT/DL"
MAX_TRAIN_SAMPLES = 12000
MAX_VAL_SAMPLES = 4000
SUBSET_ARCHIVE_NAME = "freuid_highres_subset.zip"


def run(command: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def find_train_labels(search_roots: list[Path]) -> list[Path]:
    matches: dict[str, Path] = {}
    for root in search_roots:
        if root.is_file() and root.name == "train_labels.csv":
            candidates = [root]
        elif root.exists():
            candidates = list(root.rglob("train_labels.csv"))
        else:
            candidates = []
        for candidate in candidates:
            matches[str(candidate.resolve())] = candidate
    return sorted(matches.values(), key=lambda path: str(path))


def extract_private_subset(input_root: Path, working_root: Path) -> Path | None:
    archives = sorted(input_root.rglob(SUBSET_ARCHIVE_NAME)) if input_root.exists() else []
    if not archives:
        return None
    if len(archives) != 1:
        raise FileNotFoundError(f"Expected at most one {SUBSET_ARCHIVE_NAME}, found {archives}")

    destination = working_root / "freuid-highres-subset"
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    destination_resolved = destination.resolve()
    with zipfile.ZipFile(archives[0]) as handle:
        for member in handle.infolist():
            target = (destination / member.filename).resolve()
            if target != destination_resolved and destination_resolved not in target.parents:
                raise ValueError(f"Unsafe path in {archives[0]}: {member.filename}")
        handle.extractall(destination)
    return destination


def locate_competition_root(
    input_root: Path | None = None,
    working_root: Path | None = None,
) -> tuple[Path, Path, str]:
    input_root = input_root or Path("/kaggle/input")
    working_root = working_root or Path("/kaggle/working")
    preferred = input_root / COMPETITION_SLUG
    matches = find_train_labels([preferred, input_root])
    source = "attached_input"
    if not matches:
        extracted_root = extract_private_subset(input_root, working_root)
        if extracted_root is not None:
            matches = find_train_labels([extracted_root])
            source = "private_subset_archive"
    if not matches:
        import kagglehub

        download_root = "/kaggle/temp/freuid-competition"
        previous_cache_setting = os.environ.get("DISABLE_KAGGLE_CACHE")
        os.environ["DISABLE_KAGGLE_CACHE"] = "1"
        print(
            f"FREUID input is not mounted; downloading it to Kaggle temp storage at {download_root}",
            flush=True,
        )
        try:
            attached_root = Path(
                kagglehub.competition_download(
                    COMPETITION_SLUG,
                    output_dir=download_root,
                )
            )
        finally:
            if previous_cache_setting is None:
                os.environ.pop("DISABLE_KAGGLE_CACHE", None)
            else:
                os.environ["DISABLE_KAGGLE_CACHE"] = previous_cache_setting
        print(f"kagglehub HTTP download path: {attached_root}", flush=True)
        matches = find_train_labels([attached_root, preferred, input_root])
        source = "kagglehub_http_download"
    if len(matches) != 1:
        raise FileNotFoundError(
            "Expected one FREUID train_labels.csv after mounted-input and kagglehub lookup, "
            f"found {matches}"
        )
    labels = matches[0]
    root = labels.parent
    if not (root / "train").exists():
        raise FileNotFoundError(f"Expected mounted train images under {root / 'train'}")
    return root, labels, source


def choose_holdout(labels_path: Path) -> str:
    frame = pd.read_csv(labels_path, usecols=["type", "label"])
    available = {
        str(doc_type)
        for doc_type, group in frame.groupby("type")
        if set(pd.to_numeric(group["label"], errors="raise").astype(int).unique()) == {0, 1}
    }
    if HOLDOUT_TYPE in available:
        return HOLDOUT_TYPE
    if not available:
        raise ValueError("No document type contains both labels")
    fallback = sorted(available)[0]
    print(f"Requested holdout {HOLDOUT_TYPE!r} unavailable; using {fallback!r}", flush=True)
    return fallback


def gpu_inventory() -> list[dict[str, object]]:
    return [
        {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "total_memory_bytes": int(torch.cuda.get_device_properties(index).total_memory),
        }
        for index in range(torch.cuda.device_count())
    ]


def main() -> None:
    started = time.time()
    working = Path("/kaggle/working")
    repo = working / "ai-image-forensics-comparison"
    output_dir = working / "post_freeze_dinov2b_fivecrop_loto_egypt"
    manifest_path = working / "post_freeze_highres_run_manifest.json"
    manifest: dict[str, object] = {
        "research_track": "post_freeze_highres_2026_07_13",
        "competition_eligibility": "post_freeze_research_only",
        "creates_competition_submission": False,
        "repo_ref": REPO_REF,
        "status": "starting",
    }
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Kaggle GPU accelerator was not attached")
        manifest["gpus"] = gpu_inventory()
        print(json.dumps(manifest["gpus"], indent=2), flush=True)

        run([sys.executable, "-m", "pip", "install", "--quiet", "timm>=1.0.20,<2"])
        if repo.exists():
            shutil.rmtree(repo)
        run(["git", "clone", "--depth", "1", "--branch", REPO_REF, REPO_URL, str(repo)])
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        manifest["repo_commit"] = commit

        competition_root, labels_path, data_source = locate_competition_root()
        manifest["data_source"] = data_source
        holdout_type = choose_holdout(labels_path)
        split_dir = working / "post_freeze_loto_split"
        split_dir.mkdir(parents=True, exist_ok=True)
        train_csv = split_dir / "train.csv"
        val_csv = split_dir / "val.csv"
        run(
            [
                sys.executable,
                str(repo / "scripts" / "prepare_freuid_leave_one_type_out.py"),
                "--input-csv",
                str(labels_path),
                "--holdout-type",
                holdout_type,
                "--train-out",
                str(train_csv),
                "--val-out",
                str(val_csv),
            ],
            cwd=repo,
        )

        gpu_count = torch.cuda.device_count()
        document_batch_size = 2 if gpu_count > 1 else 1
        view_batch_size = 10 if gpu_count > 1 else 5
        run(
            [
                sys.executable,
                str(repo / "scripts" / "run_freuid_timm_probe.py"),
                "--train-csv",
                str(train_csv),
                "--val-csv",
                str(val_csv),
                "--image-root",
                str(competition_root),
                "--output-dir",
                str(output_dir),
                "--model",
                "dinov2_base_518",
                "--image-size",
                "518",
                "--view-mode",
                "five_crop",
                "--five-crop-zoom",
                "1.15",
                "--batch-size",
                str(document_batch_size),
                "--view-batch-size",
                str(view_batch_size),
                "--num-workers",
                "2",
                "--device",
                "cuda",
                "--max-train-samples",
                str(MAX_TRAIN_SAMPLES),
                "--max-val-samples",
                str(MAX_VAL_SAMPLES),
                "--aggregation",
                "mean",
                "mean_max",
                "mean_max_std",
                "--primary-aggregation",
                "mean_max_std",
                "--logistic-c",
                "0.1",
                "--seed",
                "43",
            ],
            cwd=repo,
        )
        summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
        manifest.update(
            {
                "status": "complete",
                "holdout_type": holdout_type,
                "max_train_samples": MAX_TRAIN_SAMPLES,
                "max_val_samples": MAX_VAL_SAMPLES,
                "output_dir": str(output_dir),
                "primary": summary["primary"],
            }
        )
    except Exception as exc:
        manifest.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        manifest["elapsed_seconds"] = float(time.time() - started)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
