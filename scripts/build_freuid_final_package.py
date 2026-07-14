from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


RUNTIME_FILES = {
    "docker/freuid/Dockerfile": "runtime/docker/freuid/Dockerfile",
    "docker/freuid/requirements.txt": "runtime/docker/freuid/requirements.txt",
    "docker/freuid/entrypoint.sh": "runtime/docker/freuid/entrypoint.sh",
    "scripts/infer_freuid_checkpoint_ensemble.py": "runtime/scripts/infer_freuid_checkpoint_ensemble.py",
    "scripts/infer_freuid_finetune.py": "runtime/scripts/infer_freuid_finetune.py",
    "scripts/freeze_freuid_submission_artifacts.py": "runtime/scripts/freeze_freuid_submission_artifacts.py",
    "src/forensic_compare/freuid_model.py": "runtime/src/forensic_compare/freuid_model.py",
    "src/forensic_compare/freuid_transforms.py": "runtime/src/forensic_compare/freuid_transforms.py",
    "src/forensic_compare/nn_model.py": "runtime/src/forensic_compare/nn_model.py",
    "src/forensic_compare/utils.py": "runtime/src/forensic_compare/utils.py",
    "src/forensic_compare/__init__.py": "runtime/src/forensic_compare/__init__.py",
    "artifacts/freuid_2026/README.md": "runtime/artifacts/freuid_2026/README.md",
    "artifacts/freuid_2026/freeze_manifest.json": "runtime/artifacts/freuid_2026/freeze_manifest.json",
    "artifacts/freuid_2026/loto_egypt_clean_fusion_summary.json": "runtime/artifacts/freuid_2026/loto_egypt_clean_fusion_summary.json",
    "artifacts/freuid_2026/loto_egypt_screenshot_fusion_summary.json": "runtime/artifacts/freuid_2026/loto_egypt_screenshot_fusion_summary.json",
    "artifacts/freuid_2026/checkpoints/template_convnext224.pt": (
        "runtime/artifacts/freuid_2026/checkpoints/template_convnext224.pt"
    ),
    "artifacts/freuid_2026/checkpoints/forensic_efficientnet384.pt": (
        "runtime/artifacts/freuid_2026/checkpoints/forensic_efficientnet384.pt"
    ),
}

PACKAGE_FILES = {
    "work/freuid_private_final_submissions_2026_07_14/submission_private_ood_rank.csv": (
        "kaggle_submission/submission.csv"
    ),
    "work/freuid_private_final_submissions_2026_07_14/submission_private_ood_rank.lint.json": (
        "kaggle_submission/submission_lint.json"
    ),
    "work/freuid_private_final_submissions_2026_07_14/submission_private_ood_rank.submit.json": (
        "kaggle_submission/submit_manifest.json"
    ),
    "work/freuid_private_final_submissions_2026_07_14/submission_private_ood_rank.manifest.json": (
        "kaggle_submission/private_merge_manifest.json"
    ),
    "work/freuid_private_final_submissions_2026_07_14/submission_private_public_specialist.csv": (
        "kaggle_submission/public_specialist_submission.csv"
    ),
    "work/freuid_private_final_submissions_2026_07_14/submission_private_public_specialist.lint.json": (
        "kaggle_submission/public_specialist_submission_lint.json"
    ),
    "work/freuid_private_final_submissions_2026_07_14/submission_private_public_specialist.submit.json": (
        "kaggle_submission/public_specialist_submit_manifest.json"
    ),
    "work/freuid_private_final_submissions_2026_07_14/submission_private_public_specialist.manifest.json": (
        "kaggle_submission/public_specialist_private_merge_manifest.json"
    ),
    "reports/freuid_short_report_2026_07_13.md": "report/freuid_short_report_2026_07_13.md",
    "output/pdf/freuid_short_report_2026_07_13.pdf": "report/freuid_short_report_2026_07_13.pdf",
    "reports/freuid_pinned_discussion_reply_2026_07_13.md": (
        "discussion/freuid_pinned_discussion_reply_2026_07_13.md"
    ),
    "reports/freuid_code_freeze_status_2026_07_13.md": "docs/freuid_code_freeze_status_2026_07_13.md",
    "reports/freuid_reproducibility_checklist_2026_07_13.md": (
        "docs/freuid_reproducibility_checklist_2026_07_13.md"
    ),
    "reports/freuid_private_final_inference_2026_07_14.md": (
        "docs/freuid_private_final_inference_2026_07_14.md"
    ),
    "LICENSE": "LICENSE",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the frozen FREUID v3 runtime and final-package archives.")
    parser.add_argument("--runtime-zip", default="work/freuid_frozen_stack_v3_2026_07_13.zip")
    parser.add_argument("--package-zip", default="work/freuid_final_package_2026_07_13.zip")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validated_entries(mapping: dict[str, str]) -> list[dict[str, object]]:
    entries = []
    for source_value, archive_path in mapping.items():
        source = Path(source_value)
        if not source.is_file():
            raise FileNotFoundError(source)
        entries.append(
            {
                "source": str(source),
                "archive_path": archive_path,
                "bytes": int(source.stat().st_size),
                "sha256": _sha256(source),
            }
        )
    return entries


def _write_archive(path: Path, entries: list[dict[str, object]], extra_text: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for entry in entries:
            archive.write(str(entry["source"]), str(entry["archive_path"]))
        for archive_path, content in extra_text.items():
            archive.writestr(archive_path, content)


def main() -> None:
    args = parse_args()
    created_at = datetime.now(timezone.utc).isoformat()
    runtime_entries = _validated_entries(RUNTIME_FILES)
    runtime_manifest = {
        "created_at": created_at,
        "runtime": "0.85*rank(template_convnext224) + 0.15*rank(forensic_efficientnet384)",
        "kaggle_refs": ["54673713", "54673732"],
        "variants": {
            "public_specialist": {
                "kaggle_ref": "54673713",
                "submission_sha256": "f2a8737665672f2638ef88b7cdab71c00168402b21076e9383cb5c22d6ca68b2",
            },
            "ood_rank": {
                "kaggle_ref": "54673732",
                "submission_sha256": "5ce9667137ba83def3d9c139f4cd55e1d961a92c6bf42224020ec5b27b66df07",
            },
        },
        "files": runtime_entries,
    }
    runtime_readme = (
        "FREUID 2026 frozen v3 runtime\n\n"
        "Build from the repository root:\n"
        "  docker build -f runtime/docker/freuid/Dockerfile -t freuid-v3 runtime\n\n"
        "The challenge image expects /data and writes /submissions/submission.csv.\n"
        "Set FREUID_VARIANT=ood_rank (default) or FREUID_VARIANT=public_specialist.\n"
        "See runtime/artifacts/freuid_2026/freeze_manifest.json for checkpoint hashes.\n"
    )
    _write_archive(
        Path(args.runtime_zip),
        runtime_entries,
        {
            "README.txt": runtime_readme,
            "runtime_manifest.json": json.dumps(runtime_manifest, indent=2, sort_keys=True) + "\n",
        },
    )

    package_entries = _validated_entries(PACKAGE_FILES) + runtime_entries
    package_manifest = {
        "created_at": created_at,
        "competition": "the-freuid-challenge-2026-ijcai-ecai",
        "selected_submission_ref": "54673732",
        "public_specialist_ref": "54673713",
        "public_scores": {"54673732": 0.25799, "54673713": 0.25470},
        "runtime": runtime_manifest["runtime"],
        "files": package_entries,
    }
    package_readme = (
        "FREUID Challenge 2026 final package\n\n"
        "Selected OOD submission: 54673732 (public score 0.25799)\n"
        "Public specialist: 54673713 (public score 0.25470)\n\n"
        "The package contains both Kaggle CSVs, the technical report, public-source runtime, "
        "checkpoint weights, validation summaries, and Docker recipe.\n"
    )
    _write_archive(
        Path(args.package_zip),
        package_entries,
        {
            "README.txt": package_readme,
            "package_manifest.json": json.dumps(package_manifest, indent=2, sort_keys=True) + "\n",
        },
    )
    print(Path(args.runtime_zip).resolve())
    print(Path(args.package_zip).resolve())


if __name__ == "__main__":
    main()
