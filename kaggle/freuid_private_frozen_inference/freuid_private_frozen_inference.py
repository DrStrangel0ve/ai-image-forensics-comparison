from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch


REPO_URL = "https://github.com/DrStrangel0ve/ai-image-forensics-comparison.git"
REPO_REF = "freuid-final-2026-07-13"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
PUBLIC_CHECKPOINT = "template_convnext224.pt"
OOD_CHECKPOINT = "forensic_efficientnet384.pt"


def run(command: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def find_unique_file(input_root: Path, name: str) -> Path:
    matches = sorted(path for path in input_root.rglob(name) if path.is_file())
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {name!r} under {input_root}, found {matches}")
    return matches[0]


def find_private_image_root(input_root: Path) -> tuple[Path, int]:
    candidates: list[tuple[int, Path]] = []
    for directory in [input_root, *sorted(path for path in input_root.rglob("*") if path.is_dir())]:
        count = sum(
            1
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
        if count:
            candidates.append((count, directory))
    private_candidates = [
        (count, directory)
        for count, directory in candidates
        if "private_test" in {part.lower() for part in directory.parts}
    ]
    if not private_candidates:
        raise FileNotFoundError(f"No private-test images found under {input_root}")
    count, root = max(private_candidates, key=lambda item: (item[0], str(item[1])))
    return root, count


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_submission(path: Path, expected_ids: set[str]) -> dict[str, object]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or set(rows[0]) != {"id", "label"}:
        raise ValueError(f"{path} must contain exactly id,label columns and at least one row")
    ids = [str(row["id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path} contains duplicate ids")
    if set(ids) != expected_ids:
        missing = sorted(expected_ids - set(ids))[:5]
        extra = sorted(set(ids) - expected_ids)[:5]
        raise ValueError(f"{path} id mismatch; missing={missing} extra={extra}")
    scores = [float(row["label"]) for row in rows]
    if not all(math.isfinite(score) and 0.0 <= score <= 1.0 for score in scores):
        raise ValueError(f"{path} contains invalid probability scores")
    if len(set(scores)) < 2:
        raise ValueError(f"{path} contains a constant score")
    return {
        "path": str(path),
        "rows": len(rows),
        "sha256": sha256(path),
        "score_min": min(scores),
        "score_max": max(scores),
        "unique_scores": len(set(scores)),
    }


def inference_commands(
    repo: Path,
    image_root: Path,
    public_checkpoint: Path,
    ood_checkpoint: Path,
    working: Path,
) -> dict[str, list[str]]:
    common = [
        "--input-dir",
        str(image_root),
        "--recursive",
        "--batch-size",
        "64",
        "--num-workers",
        "4",
        "--device",
        "cuda",
    ]
    return {
        "public_specialist": [
            sys.executable,
            str(repo / "scripts" / "infer_freuid_finetune.py"),
            "--checkpoint",
            str(public_checkpoint),
            "--output-csv",
            str(working / "private_public_specialist.csv"),
            "--manifest-out",
            str(working / "private_public_specialist.manifest.json"),
            *common,
        ],
        "ood_rank": [
            sys.executable,
            str(repo / "scripts" / "infer_freuid_checkpoint_ensemble.py"),
            "--checkpoint",
            str(public_checkpoint),
            "--checkpoint",
            str(ood_checkpoint),
            "--weight",
            "0.85",
            "--weight",
            "0.15",
            "--normalization",
            "rank",
            "--output-csv",
            str(working / "private_ood_rank.csv"),
            "--manifest-out",
            str(working / "private_ood_rank.manifest.json"),
            *common,
        ],
    }


def execution_batches(variants: list[str], gpu_count: int) -> list[list[tuple[str, int]]]:
    if gpu_count < 1:
        raise ValueError("gpu_count must be positive")
    return [
        [
            (variant, batch_index)
            for batch_index, variant in enumerate(variants[start : start + gpu_count])
        ]
        for start in range(0, len(variants), gpu_count)
    ]


def main() -> None:
    started = time.time()
    input_root = Path("/kaggle/input")
    working = Path("/kaggle/working")
    repo = Path("/tmp/ai-image-forensics-comparison")
    run_manifest_path = working / "private_frozen_inference_manifest.json"
    manifest: dict[str, object] = {
        "competition_eligibility": "frozen_final_inference_only",
        "creates_competition_submission": False,
        "repo_ref": REPO_REF,
        "status": "starting",
    }
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Kaggle GPU accelerator was not attached")
        gpus = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "total_memory_bytes": int(torch.cuda.get_device_properties(index).total_memory),
            }
            for index in range(torch.cuda.device_count())
        ]
        manifest["gpus"] = gpus
        print(json.dumps(gpus, indent=2), flush=True)

        if repo.exists():
            shutil.rmtree(repo)
        run(["git", "clone", "--depth", "1", "--branch", REPO_REF, REPO_URL, str(repo)])
        manifest["repo_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
        image_root, image_count = find_private_image_root(input_root)
        expected_ids = {
            path.stem
            for path in image_root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        }
        public_checkpoint = find_unique_file(input_root, PUBLIC_CHECKPOINT)
        ood_checkpoint = find_unique_file(input_root, OOD_CHECKPOINT)
        commands = inference_commands(
            repo,
            image_root,
            public_checkpoint,
            ood_checkpoint,
            working,
        )

        return_codes: dict[str, int] = {}
        for batch in execution_batches(list(commands), len(gpus)):
            processes: dict[str, subprocess.Popen[str]] = {}
            for variant, gpu_index in batch:
                command = commands[variant]
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
                print(f"[{variant}] + {' '.join(command)}", flush=True)
                processes[variant] = subprocess.Popen(
                    command,
                    cwd=repo,
                    env=environment,
                    text=True,
                )
            return_codes.update(
                {variant: process.wait() for variant, process in processes.items()}
            )
        failures = {variant: code for variant, code in return_codes.items() if code != 0}
        if failures:
            raise RuntimeError(f"Frozen inference subprocess failures: {failures}")

        outputs = {
            "public_specialist": validate_submission(
                working / "private_public_specialist.csv", expected_ids
            ),
            "ood_rank": validate_submission(working / "private_ood_rank.csv", expected_ids),
        }
        manifest.update(
            {
                "status": "complete",
                "image_root": str(image_root),
                "image_count": image_count,
                "outputs": outputs,
                "checkpoints": {
                    "public_specialist": {
                        "path": str(public_checkpoint),
                        "sha256": sha256(public_checkpoint),
                    },
                    "ood_rank_secondary": {
                        "path": str(ood_checkpoint),
                        "sha256": sha256(ood_checkpoint),
                    },
                },
            }
        )
    except Exception as exc:
        manifest.update({"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        manifest["elapsed_seconds"] = float(time.time() - started)
        run_manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
