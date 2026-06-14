from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


DEFAULT_SCAN_FILES = [
    "README.md",
    "reports/submission_text_drafts_2026_06_14.md",
    "reports/paper_section_drafts_2026_06_14.md",
    "reports/submission_packet_2026_06_13.md",
    "reports/external_benchmark_readiness_2026_06_14.md",
    "reports/assets/paper_skeletons/wifs_2026_paper_skeleton.tex",
    "reports/assets/paper_skeletons/dff_2026_workshop_skeleton.tex",
]

BENCHMARK_PATTERN = re.compile(r"\b(?:NTIRE|ImageCLEF)\b", flags=re.IGNORECASE)
CLAIM_PATTERN = re.compile(
    r"\b(?:official\s+(?:score|auc|accuracy|result)|leaderboard|ranked?|submitted|participated|won|placed)\b",
    flags=re.IGNORECASE,
)
NEGATING_PATTERN = re.compile(
    r"\b(?:no|not|none|absent|closed_not_submitted|closed|proxy|style|inspired|deadline|watch)\b",
    flags=re.IGNORECASE,
)

REQUIRED_READINESS_PHRASES = [
    "not as official scored submissions",
    "proxy robustness evidence",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lint publication-facing files for unsafe NTIRE/ImageCLEF official-score claims."
    )
    parser.add_argument("--repo-root", default=".", help="Repository root for validating relative paths.")
    parser.add_argument(
        "--scan-files",
        nargs="*",
        default=DEFAULT_SCAN_FILES,
        help="Relative files to scan for unsafe external-benchmark claims.",
    )
    parser.add_argument(
        "--readiness-status",
        default="reports/assets/external_benchmark_readiness.csv",
        help="Machine-readable external benchmark readiness status table.",
    )
    parser.add_argument(
        "--readiness-report",
        default="reports/external_benchmark_readiness_2026_06_14.md",
        help="Markdown external benchmark readiness report.",
    )
    parser.add_argument(
        "--out-path",
        default="reports/external_benchmark_claim_lint_2026_06_14.md",
        help="Markdown lint report to write.",
    )
    parser.add_argument(
        "--checks-out",
        default="reports/assets/external_benchmark_claim_lint.csv",
        help="Machine-readable lint checks to write.",
    )
    return parser.parse_args()


def _add_check(rows: list[dict[str, object]], check: str, passed: bool, detail: str) -> None:
    rows.append({"check": check, "passed": bool(passed), "detail": detail})


def _relative_path(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _unsafe_claims(path: Path, repo_root: Path) -> list[str]:
    issues = []
    relative = _relative_path(path, repo_root)
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not BENCHMARK_PATTERN.search(line) or not CLAIM_PATTERN.search(line):
            continue
        if NEGATING_PATTERN.search(line):
            continue
        issues.append(f"{relative}:{line_number}: {line.strip()}")
    return issues


def lint_external_benchmark_claims(
    repo_root: Path,
    scan_files: list[str],
    readiness_status: Path,
    readiness_report: Path,
) -> pd.DataFrame:
    repo_root = repo_root.resolve()
    rows: list[dict[str, object]] = []

    missing_files = []
    unsafe_claims = []
    for relative in scan_files:
        path = repo_root / relative
        if not path.exists():
            missing_files.append(relative)
            continue
        unsafe_claims.extend(_unsafe_claims(path, repo_root))

    _add_check(
        rows,
        "scan files exist",
        not missing_files,
        "all scan files present" if not missing_files else ", ".join(missing_files),
    )
    _add_check(
        rows,
        "no unsafe official NTIRE/ImageCLEF claims",
        not unsafe_claims,
        "no unsafe claims found" if not unsafe_claims else " | ".join(unsafe_claims),
    )

    status_path = repo_root / readiness_status
    status_exists = status_path.exists()
    _add_check(
        rows,
        "readiness status CSV exists",
        status_exists,
        readiness_status.as_posix() if status_exists else "missing",
    )
    if status_exists:
        status = pd.read_csv(status_path)
        required_ids = {"ntire_2026_robust_aigc", "imageclef_2026_deepfake"}
        present_ids = set(status["benchmark_id"])
        _add_check(
            rows,
            "readiness rows cover NTIRE and ImageCLEF",
            required_ids <= present_ids,
            ", ".join(sorted(present_ids)),
        )
        _add_check(
            rows,
            "official scores remain none",
            set(status["official_score"].astype(str)) == {"none"},
            ", ".join(sorted(set(status["official_score"].astype(str)))),
        )
        _add_check(
            rows,
            "official status remains closed_not_submitted",
            set(status["official_status"].astype(str)) == {"closed_not_submitted"},
            ", ".join(sorted(set(status["official_status"].astype(str)))),
        )

    report_path = repo_root / readiness_report
    report_exists = report_path.exists()
    _add_check(
        rows,
        "readiness report exists",
        report_exists,
        readiness_report.as_posix() if report_exists else "missing",
    )
    if report_exists:
        text = report_path.read_text(encoding="utf-8")
        for phrase in REQUIRED_READINESS_PHRASES:
            _add_check(
                rows,
                f"readiness report states: {phrase}",
                phrase in text,
                "present" if phrase in text else "missing",
            )

    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def write_report(frame: pd.DataFrame, out_path: Path) -> None:
    passed = int(frame["passed"].sum())
    total = len(frame)
    status = "PASS" if passed == total else "FAIL"
    lines = [
        "# External Benchmark Claim Lint",
        "",
        "Run date: 2026-06-14",
        "",
        f"Status: **{status}** ({passed}/{total} checks passed).",
        "",
        "Generated by `scripts/lint_external_benchmark_claims.py` to prevent NTIRE/ImageCLEF proxy evidence from being described as official challenge performance.",
        "",
        _markdown_table(frame),
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    frame = lint_external_benchmark_claims(
        repo_root=Path(args.repo_root),
        scan_files=list(args.scan_files),
        readiness_status=Path(args.readiness_status),
        readiness_report=Path(args.readiness_report),
    )
    checks_path = Path(args.checks_out)
    checks_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(checks_path, index=False)
    write_report(frame, Path(args.out_path))
    if not bool(frame["passed"].all()):
        failed = frame[~frame["passed"]]
        raise SystemExit(
            "External benchmark claim lint failed: "
            + "; ".join(f"{row.check} ({row.detail})" for row in failed.itertuples(index=False))
        )
    print(Path(args.out_path).resolve())
    print(checks_path.resolve())


if __name__ == "__main__":
    main()
