from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forensic_compare.utils import write_json


DEFAULT_COMPETITION = "the-freuid-challenge-2026-ijcai-ecai"
DEFAULT_KERNEL = "arnavmalani/freuid-photometric-offline-submission"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check FREUID Kaggle release readiness without creating a submission."
    )
    parser.add_argument("--competition", default=DEFAULT_COMPETITION)
    parser.add_argument("--kernel", default=DEFAULT_KERNEL)
    parser.add_argument(
        "--sample-submission",
        default="data/raw/freuid_2026/small_files/sample_submission.csv",
        help="Local sample_submission.csv used to count required public-test ids.",
    )
    parser.add_argument(
        "--public-image-dir",
        default="data/raw/freuid_2026/images/public_test",
        help="Directory containing acquired public-test images.",
    )
    parser.add_argument(
        "--candidate-submission",
        default="outputs/freuid_2026/freuid_public_fused_submission.csv",
        help="Expected real candidate submission path.",
    )
    parser.add_argument(
        "--kernel-source-required",
        action="store_true",
        help="Fail readiness when the pushed Kaggle kernel does not show the competition source attached.",
    )
    parser.add_argument(
        "--manifest-out",
        default="outputs/freuid_2026/release_state_manifest.json",
        help="JSON readiness report.",
    )
    return parser.parse_args()


def _plain(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if hasattr(value, "to_dict"):
        return _plain(value.to_dict())
    if hasattr(value, "__dict__"):
        return {
            str(key): _plain(item)
            for key, item in value.__dict__.items()
            if not key.startswith("_")
        }
    return str(value)


def _safe_step(name: str, func) -> tuple[dict[str, object], Any | None]:
    try:
        value = func()
        return {"name": name, "status": "ok"}, value
    except Exception as exc:  # noqa: BLE001 - this is a diagnostic script.
        return {"name": name, "status": "error", "error": f"{type(exc).__name__}: {exc}"}, None


def _image_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob("*") if item.suffix.lower() in IMAGE_EXTENSIONS)


def _submission_summary(path: Path, sample_path: Path) -> dict[str, object]:
    if not path.exists():
        return {"exists": False}
    submission = pd.read_csv(path)
    summary: dict[str, object] = {
        "exists": True,
        "path": str(path),
        "columns": list(submission.columns),
        "rows": int(len(submission)),
    }
    if "label" in submission.columns:
        labels = pd.to_numeric(submission["label"], errors="coerce")
        valid = labels.notna().all() and set(labels.astype(int).unique()).issubset({0, 1})
        summary["binary_labels"] = bool(valid)
        if valid:
            summary["label_counts"] = {
                str(key): int(value)
                for key, value in labels.astype(int).value_counts().sort_index().items()
            }
    if sample_path.exists():
        sample = pd.read_csv(sample_path)
        summary["matches_sample_rows"] = int(len(sample)) == int(len(submission))
        if "id" in sample.columns and "id" in submission.columns:
            summary["matches_sample_ids"] = set(sample["id"].astype(str)) == set(
                submission["id"].astype(str)
            )
            summary["matches_sample_order"] = sample["id"].astype(str).tolist() == submission[
                "id"
            ].astype(str).tolist()
    return summary


def _load_kaggle_api():
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    return api


def _kernel_metadata(api, kernel: str) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="freuid_kernel_meta_") as tmp:
        pulled = api.kernels_pull(kernel, path=tmp, metadata=True, quiet=True)
        metadata_path = Path(pulled) / "kernel-metadata.json"
        with metadata_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)


def build_report(args: argparse.Namespace) -> dict[str, object]:
    sample_path = Path(args.sample_submission)
    public_dir = Path(args.public_image_dir)
    candidate_path = Path(args.candidate_submission)

    report: dict[str, object] = {
        "competition": args.competition,
        "kernel": args.kernel,
        "local": {
            "sample_submission_path": str(sample_path),
            "sample_submission_exists": sample_path.exists(),
            "sample_submission_rows": None,
            "public_image_dir": str(public_dir),
            "public_images_acquired": _image_count(public_dir),
            "candidate_submission": _submission_summary(candidate_path, sample_path),
        },
        "kaggle": {},
        "checks": [],
        "ready_to_submit": False,
        "recommended_next_action": None,
    }

    if sample_path.exists():
        report["local"]["sample_submission_rows"] = int(len(pd.read_csv(sample_path)))  # type: ignore[index]

    step, api = _safe_step("kaggle_auth", _load_kaggle_api)
    report["checks"].append(step)  # type: ignore[union-attr]
    if api is None:
        report["recommended_next_action"] = "Fix Kaggle authentication before attempting release."
        return report

    step, competitions = _safe_step(
        "competition_lookup",
        lambda: api.competitions_list(search=args.competition, page_size=5),
    )
    report["checks"].append(step)  # type: ignore[union-attr]
    if competitions is not None:
        competitions_plain = _plain(competitions)
        report["kaggle"]["competition_lookup"] = competitions_plain  # type: ignore[index]

    step, submissions = _safe_step(
        "submission_lookup",
        lambda: api.competition_submissions(args.competition, page_size=10),
    )
    report["checks"].append(step)  # type: ignore[union-attr]
    submissions_plain = _plain(submissions) if submissions is not None else []
    report["kaggle"]["submissions"] = submissions_plain  # type: ignore[index]
    report["kaggle"]["submission_count"] = len(submissions or [])  # type: ignore[index]

    step, status = _safe_step("kernel_status", lambda: api.kernels_status(args.kernel))
    report["checks"].append(step)  # type: ignore[union-attr]
    report["kaggle"]["kernel_status"] = _plain(status)  # type: ignore[index]

    step, metadata = _safe_step("kernel_metadata_pull", lambda: _kernel_metadata(api, args.kernel))
    report["checks"].append(step)  # type: ignore[union-attr]
    metadata = metadata or {}
    report["kaggle"]["kernel_metadata"] = metadata  # type: ignore[index]

    competition_sources = metadata.get("competition_sources", []) if isinstance(metadata, dict) else []
    report["kaggle"]["kernel_competition_source_attached"] = args.competition in competition_sources  # type: ignore[index]

    candidate = report["local"]["candidate_submission"]  # type: ignore[index]
    candidate_ready = bool(
        isinstance(candidate, dict)
        and candidate.get("exists")
        and candidate.get("matches_sample_rows")
        and candidate.get("matches_sample_ids")
        and candidate.get("binary_labels", True)
    )
    kernel_source_ready = bool(report["kaggle"]["kernel_competition_source_attached"])  # type: ignore[index]
    report["ready_to_submit"] = candidate_ready and (
        kernel_source_ready or not args.kernel_source_required
    )

    if candidate_ready:
        report["recommended_next_action"] = "Lint and submit the candidate CSV."
    elif not kernel_source_ready:
        report["recommended_next_action"] = (
            "Attach the FREUID competition data to the Kaggle notebook in the web UI, then rerun it."
        )
    else:
        report["recommended_next_action"] = (
            "Generate public-test predictions before attempting a leaderboard submission."
        )
    return report


def main() -> None:
    args = parse_args()
    report = build_report(args)
    manifest_path = Path(args.manifest_out)
    write_json(report, manifest_path)
    print(manifest_path.resolve())
    if not report["ready_to_submit"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
