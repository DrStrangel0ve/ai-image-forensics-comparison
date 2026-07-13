from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and smoke-test the FREUID Docker inference image with no network."
    )
    parser.add_argument("--tag", default="freuid-frozen-stack:local")
    parser.add_argument("--dockerfile", default="docker/freuid/Dockerfile")
    parser.add_argument("--context", default=".")
    parser.add_argument("--image-dir", default="data/raw/freuid_2026/images/public_test/public_test")
    parser.add_argument("--image-count", type=int, default=5)
    parser.add_argument("--output-dir", default="outputs/freuid_2026/docker_smoke")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--variant", choices=["ood_rank", "public_specialist"], default="ood_rank")
    parser.add_argument("--manifest-out", default=None)
    parser.add_argument("--readiness-timeout-seconds", type=int, default=60)
    parser.add_argument("--build-timeout-seconds", type=int, default=1800)
    parser.add_argument("--run-timeout-seconds", type=int, default=600)
    return parser.parse_args()


def _run(command: list[str], cwd: Path, timeout_seconds: int) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        return subprocess.CompletedProcess(
            args=command,
            returncode=124,
            stdout=exc.stdout or "",
            stderr=f"Command timed out after {timeout_seconds} seconds",
        )


def _require_success(result: subprocess.CompletedProcess[str], action: str) -> None:
    if result.returncode != 0:
        message = [
            f"{action} failed with exit code {result.returncode}",
            "STDOUT:",
            result.stdout.strip(),
            "STDERR:",
            result.stderr.strip(),
        ]
        raise RuntimeError("\n".join(message))


def _check_docker(cwd: Path, timeout_seconds: int) -> str:
    result = _run(["docker", "info", "--format", "{{.ServerVersion}}"], cwd, timeout_seconds)
    _require_success(result, "docker readiness check")
    return result.stdout.strip()


def _stage_images(source_dir: Path, smoke_data_dir: Path, image_count: int) -> list[str]:
    source_paths = sorted(
        path for path in source_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if len(source_paths) < image_count:
        raise FileNotFoundError(f"Need {image_count} smoke images, found {len(source_paths)} in {source_dir}")
    smoke_data_dir.mkdir(parents=True, exist_ok=True)
    for old_file in smoke_data_dir.iterdir():
        if old_file.is_file():
            old_file.unlink()
    selected = source_paths[:image_count]
    for path in selected:
        shutil.copy2(path, smoke_data_dir / path.name)
    return [path.stem for path in selected]


def _validate_submission(path: Path, expected_ids: list[str]) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["id", "label"]:
            raise ValueError(f"Unexpected submission columns: {reader.fieldnames}")
        rows = list(reader)
    ids = [row["id"] for row in rows]
    if ids != expected_ids:
        raise ValueError(f"Unexpected output IDs: expected {expected_ids}, got {ids}")
    scores = [float(row["label"]) for row in rows]
    bad_scores = [score for score in scores if not 0.0 <= score <= 1.0]
    if bad_scores:
        raise ValueError(f"Found out-of-range scores: {bad_scores[:5]}")
    return {
        "rows": len(rows),
        "score_min": min(scores),
        "score_max": max(scores),
        "score_mean": sum(scores) / len(scores),
    }


def main() -> None:
    args = parse_args()
    root = Path.cwd()
    output_dir = Path(args.output_dir)
    smoke_data_dir = output_dir / "data"
    submission_dir = output_dir / "submissions"
    submission_dir.mkdir(parents=True, exist_ok=True)
    expected_ids = _stage_images(Path(args.image_dir), smoke_data_dir, args.image_count)
    docker_version = _check_docker(root, args.readiness_timeout_seconds)

    if not args.skip_build:
        build = _run(
            ["docker", "build", "-f", args.dockerfile, "-t", args.tag, args.context],
            root,
            args.build_timeout_seconds,
        )
        _require_success(build, "docker build")

    run = _run(
        [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "-v",
            f"{smoke_data_dir.resolve()}:/data:ro",
            "-v",
            f"{submission_dir.resolve()}:/submissions",
            "-e",
            f"FREUID_VARIANT={args.variant}",
            args.tag,
        ],
        root,
        args.run_timeout_seconds,
    )
    _require_success(run, "docker no-network smoke run")
    validation = _validate_submission(submission_dir / "submission.csv", expected_ids)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "docker_server_version": docker_version,
        "tag": args.tag,
        "dockerfile": args.dockerfile,
        "image_dir": args.image_dir,
        "image_count": args.image_count,
        "output_dir": str(output_dir),
        "skip_build": bool(args.skip_build),
        "variant": args.variant,
        "validation": validation,
    }
    manifest_path = Path(args.manifest_out) if args.manifest_out else output_dir / "docker_smoke_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(manifest_path.resolve())
    print(f"rows={validation['rows']} score_mean={validation['score_mean']:.6f}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
