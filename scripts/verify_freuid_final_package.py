from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from forensic_compare.utils import write_json
from lint_freuid_submission import lint_freuid_submission


REQUIRED_PACKAGE_ENTRIES = [
    "README.txt",
    "package_manifest.json",
    "kaggle_submission/submission.csv",
    "kaggle_submission/submission_lint.json",
    "kaggle_submission/submit_manifest.json",
    "report/freuid_short_report_draft_2026_07_10.pdf",
    "report/freuid_short_report_draft_2026_07_10.md",
    "discussion/freuid_pinned_discussion_reply_draft_2026_07_10.md",
    "docs/freuid_release_runbook.md",
    "docs/freuid_reproducibility_checklist_2026_07_10.md",
]

REQUIRED_RELEASE_ASSETS = [
    "freuid_final_package_draft_2026_07_10.zip",
    "freuid_frozen_stack_2026_07_10.zip",
    "freuid_short_report_draft_2026_07_10.pdf",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify FREUID final-package draft readiness.")
    parser.add_argument("--package-zip", default="work/freuid_final_package_draft_2026_07_10.zip")
    parser.add_argument("--sample-submission", default="data/raw/freuid_2026/small_files/sample_submission.csv")
    parser.add_argument(
        "--submission",
        default="outputs/freuid_2026/public_12k_fourway_fusion_submission_packaged/submission.csv",
    )
    parser.add_argument("--report-pdf", default="output/pdf/freuid_short_report_draft_2026_07_10.pdf")
    parser.add_argument(
        "--discussion-draft",
        default="reports/freuid_pinned_discussion_reply_draft_2026_07_10.md",
    )
    parser.add_argument("--release-tag", default="freuid-freeze-2026-07-10")
    parser.add_argument("--expected-submission-ref", default="54511333")
    parser.add_argument("--expected-public-score", default="0.37009")
    parser.add_argument(
        "--manifest-out",
        default="outputs/freuid_2026/final_package_verification_manifest.json",
    )
    parser.add_argument("--skip-release-check", action="store_true")
    return parser.parse_args()


def _check(name: str, passed: bool, detail: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    result: dict[str, Any] = {"check": name, "passed": bool(passed), "detail": detail}
    if extra:
        result.update(extra)
    return result


def _verify_package_zip(path: Path) -> tuple[bool, dict[str, Any]]:
    if not path.exists():
        return False, {"detail": f"missing package zip: {path}"}
    with zipfile.ZipFile(path) as archive:
        bad_entry = archive.testzip()
        names = set(archive.namelist())
        missing = [name for name in REQUIRED_PACKAGE_ENTRIES if name not in names]
        package_manifest = json.loads(archive.read("package_manifest.json").decode("utf-8"))
    passed = bad_entry is None and not missing
    return passed, {
        "detail": "package zip contains required entries and has no corrupt members"
        if passed
        else f"bad_entry={bad_entry}, missing={missing}",
        "bytes": int(path.stat().st_size),
        "required_entries": REQUIRED_PACKAGE_ENTRIES,
        "package_manifest": package_manifest,
    }


def _verify_submission(sample_path: Path, submission_path: Path) -> tuple[bool, dict[str, Any]]:
    manifest_path = submission_path.with_suffix(".final_package_verify_lint.json")
    passed, report = lint_freuid_submission(
        sample_submission_path=sample_path,
        submission_path=submission_path,
        manifest_path=manifest_path,
        allow_reordered_ids=False,
        allow_score_labels=True,
    )
    return passed, {
        "detail": "submission lint passed" if passed else "submission lint failed",
        "lint_manifest_path": str(manifest_path),
        "n_rows": report.get("n_submission_rows"),
        "checks": report.get("checks", []),
    }


def _verify_report_pdf(path: Path) -> tuple[bool, dict[str, Any]]:
    if not path.exists():
        return False, {"detail": f"missing report PDF: {path}"}
    with path.open("rb") as handle:
        header = handle.read(5)
    passed = header == b"%PDF-"
    return passed, {
        "detail": "report PDF exists and has a PDF header" if passed else f"unexpected PDF header: {header!r}",
        "bytes": int(path.stat().st_size),
    }


def _verify_discussion_draft(path: Path, expected_submission_ref: str, expected_public_score: str) -> tuple[bool, dict[str, Any]]:
    if not path.exists():
        return False, {"detail": f"missing discussion draft: {path}"}
    text = path.read_text(encoding="utf-8")
    required_snippets = [
        "https://github.com/DrStrangel0ve/ai-image-forensics-comparison",
        "freuid-freeze-2026-07-10",
        expected_submission_ref,
        expected_public_score,
    ]
    missing = [snippet for snippet in required_snippets if snippet not in text]
    return not missing, {
        "detail": "discussion draft contains repo, release, submission ref, and score"
        if not missing
        else f"missing snippets: {missing}",
        "required_snippets": required_snippets,
    }


def _verify_release(tag: str) -> tuple[bool, dict[str, Any]]:
    result = subprocess.run(
        ["gh", "release", "view", tag, "--json", "url,assets"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return False, {"detail": result.stderr.strip() or result.stdout.strip()}
    payload = json.loads(result.stdout)
    assets = {asset["name"]: asset for asset in payload.get("assets", [])}
    missing = [name for name in REQUIRED_RELEASE_ASSETS if name not in assets]
    return not missing, {
        "detail": "release contains required assets" if not missing else f"missing release assets: {missing}",
        "release_url": payload.get("url"),
        "assets": {
            name: {
                "size": assets[name].get("size"),
                "digest": assets[name].get("digest"),
                "url": assets[name].get("url"),
            }
            for name in sorted(assets)
        },
    }


def main() -> None:
    args = parse_args()
    checks: list[dict[str, Any]] = []

    package_passed, package_info = _verify_package_zip(Path(args.package_zip))
    checks.append(_check("package_zip", package_passed, str(package_info.pop("detail")), package_info))

    submission_passed, submission_info = _verify_submission(Path(args.sample_submission), Path(args.submission))
    checks.append(_check("submission_lint", submission_passed, str(submission_info.pop("detail")), submission_info))

    report_passed, report_info = _verify_report_pdf(Path(args.report_pdf))
    checks.append(_check("report_pdf", report_passed, str(report_info.pop("detail")), report_info))

    discussion_passed, discussion_info = _verify_discussion_draft(
        Path(args.discussion_draft),
        expected_submission_ref=args.expected_submission_ref,
        expected_public_score=args.expected_public_score,
    )
    checks.append(_check("discussion_draft", discussion_passed, str(discussion_info.pop("detail")), discussion_info))

    if not args.skip_release_check:
        release_passed, release_info = _verify_release(args.release_tag)
        checks.append(_check("release_assets", release_passed, str(release_info.pop("detail")), release_info))

    passed = all(bool(check["passed"]) for check in checks)
    manifest = {
        "status": "pass" if passed else "fail",
        "release_tag": args.release_tag,
        "expected_submission_ref": args.expected_submission_ref,
        "expected_public_score": args.expected_public_score,
        "checks": checks,
    }
    write_json(manifest, args.manifest_out)
    print(Path(args.manifest_out).resolve())
    print(manifest["status"])
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
