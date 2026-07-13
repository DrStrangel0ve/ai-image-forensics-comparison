from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch


REPO_URL = "https://github.com/DrStrangel0ve/ai-image-forensics-comparison.git"
REPO_REF = "freuid-final-2026-07-13"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
PUBLIC_CHECKPOINT = "template_convnext224.pt"
OOD_CHECKPOINT = "forensic_efficientnet384.pt"
PUBLIC_WEIGHT = 0.85
FORENSIC_WEIGHT = 0.15


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


def materialize_private_image_root(input_root: Path, scratch_root: Path) -> tuple[Path, int]:
    try:
        return find_private_image_root(input_root)
    except FileNotFoundError as original_error:
        archives = sorted(path for path in input_root.rglob("private_test.tar") if path.is_file())
        if len(archives) != 1:
            raise original_error

    scratch_root.mkdir(parents=True, exist_ok=True)
    resolved_scratch = scratch_root.resolve()
    with tarfile.open(archives[0], mode="r") as archive:
        members = archive.getmembers()
        for member in members:
            target = (scratch_root / member.name).resolve()
            if not target.is_relative_to(resolved_scratch) or member.issym() or member.islnk():
                raise ValueError(f"Unsafe private-test TAR member: {member.name}")
        archive.extractall(scratch_root, members=members)
    return find_private_image_root(scratch_root)


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


def member_commands(
    repo: Path,
    image_root: Path,
    public_checkpoint: Path,
    ood_checkpoint: Path,
    working: Path,
) -> dict[str, list[str]]:
    common = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--score-member",
        "--repo",
        str(repo),
        "--input-dir",
        str(image_root),
        "--output-dir",
        str(working),
        "--recursive",
        "--batch-size",
        "64",
        "--num-workers",
        "4",
        "--device",
        "cuda",
    ]
    return {
        "public_member": [
            *common,
            "--member-name",
            "public_member",
            "--checkpoint",
            str(public_checkpoint),
        ],
        "forensic_member": [
            *common,
            "--member-name",
            "forensic_member",
            "--checkpoint",
            str(ood_checkpoint),
        ],
    }


def load_member_scores(path: Path) -> tuple[list[str], np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        ids = payload["ids"].astype(str).tolist()
        scores = np.asarray(payload["scores"], dtype=float)
    if scores.ndim != 1 or len(ids) != scores.size:
        raise ValueError(f"Invalid member score payload: {path}")
    if not np.isfinite(scores).all() or not ((0.0 <= scores) & (scores <= 1.0)).all():
        raise ValueError(f"Invalid member scores: {path}")
    return ids, scores


def fuse_ranked_scores(
    public_scores: np.ndarray,
    forensic_scores: np.ndarray,
    ranker: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    if public_scores.shape != forensic_scores.shape:
        raise ValueError("Frozen member score vectors must have the same shape")
    ranked_public = ranker(public_scores)
    ranked_forensic = ranker(forensic_scores)
    return PUBLIC_WEIGHT * ranked_public + FORENSIC_WEIGHT * ranked_forensic


def write_score_csv(path: Path, ids: list[str], scores: np.ndarray) -> None:
    if len(ids) != scores.size:
        raise ValueError("Score CSV ids and scores must have the same length")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "label"])
        writer.writeheader()
        for image_id, score in zip(ids, scores):
            writer.writerow({"id": image_id, "label": f"{score:.10f}"})


def score_member_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-member", action="store_true", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--member-name", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo)
    sys.path.insert(0, str(repo / "src"))
    sys.path.insert(0, str(repo / "scripts"))
    from forensic_compare.utils import resolve_device
    from infer_freuid_finetune import _image_paths, score_checkpoint

    checkpoint = Path(args.checkpoint)
    paths = _image_paths(Path(args.input_dir), args.recursive, 0)
    ids, scores, metadata = score_checkpoint(
        checkpoint,
        paths,
        device=resolve_device("cuda"),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    output_dir = Path(args.output_dir)
    npz_path = output_dir / f"{args.member_name}.npz"
    manifest_path = output_dir / f"{args.member_name}.manifest.json"
    np.savez(npz_path, ids=np.asarray(ids, dtype=str), scores=np.asarray(scores, dtype=float))
    member_manifest = {
        **metadata,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "member_name": args.member_name,
        "n_images": len(ids),
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "score_mean": float(np.mean(scores)),
        "score_payload": str(npz_path),
    }
    manifest_path.write_text(
        json.dumps(member_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(member_manifest, indent=2, sort_keys=True), flush=True)


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
        image_root, image_count = materialize_private_image_root(
            input_root, Path("/tmp/freuid-private-input")
        )
        expected_ids = {
            path.stem
            for path in image_root.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        }
        public_checkpoint = find_unique_file(input_root, PUBLIC_CHECKPOINT)
        ood_checkpoint = find_unique_file(input_root, OOD_CHECKPOINT)
        commands = member_commands(
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

        public_ids, public_scores = load_member_scores(working / "public_member.npz")
        forensic_ids, forensic_scores = load_member_scores(working / "forensic_member.npz")
        if public_ids != forensic_ids:
            raise ValueError("Frozen member score payloads returned a different id order")
        if set(public_ids) != expected_ids:
            raise ValueError("Frozen member score payload ids do not match private images")
        sys.path.insert(0, str(repo / "scripts"))
        from infer_freuid_checkpoint_ensemble import rank_percentiles

        ood_scores = fuse_ranked_scores(public_scores, forensic_scores, rank_percentiles)
        public_output = working / "private_public_specialist.csv"
        ood_output = working / "private_ood_rank.csv"
        write_score_csv(public_output, public_ids, public_scores)
        write_score_csv(ood_output, public_ids, ood_scores)
        (working / "private_public_specialist.manifest.json").write_text(
            json.dumps(
                {
                    "runtime": "shared_frozen_member_scores",
                    "member": "public_member",
                    "n_images": len(public_ids),
                    "output_csv": str(public_output),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        (working / "private_ood_rank.manifest.json").write_text(
            json.dumps(
                {
                    "runtime": "shared_frozen_member_scores",
                    "normalization": "rank",
                    "members": ["public_member", "forensic_member"],
                    "weights": [PUBLIC_WEIGHT, FORENSIC_WEIGHT],
                    "n_images": len(public_ids),
                    "output_csv": str(ood_output),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        outputs = {
            "public_specialist": validate_submission(public_output, expected_ids),
            "ood_rank": validate_submission(ood_output, expected_ids),
        }
        manifest.update(
            {
                "status": "complete",
                "image_root": str(image_root),
                "image_count": image_count,
                "inference_runtime": "shared_frozen_member_scores",
                "fusion_weights": [PUBLIC_WEIGHT, FORENSIC_WEIGHT],
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
    if "--score-member" in sys.argv:
        score_member_main()
    else:
        main()
