from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path

import pandas as pd


DEFAULT_RUN_DATE = date.today()
DEFAULT_ARTIFACTS = [
    "reports/assets/paper_skeletons/wifs_2026_paper_skeleton.tex",
    "reports/assets/paper_skeletons/dff_2026_workshop_skeleton.tex",
    "reports/wifs_manuscript_draft_2026_06_14.md",
    "reports/dff_manuscript_draft_2026_06_14.md",
    "reports/paper_section_drafts_2026_06_14.md",
    "reports/submission_text_drafts_2026_06_14.md",
]

AUTHOR_FIELD = re.compile(r"\\author\{(?P<author>[^}]*)\}")
EMAIL_ADDRESS = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
GITHUB_URL = re.compile(r"https?://(?:www\.)?github\.com/[^\s)\]}]+", re.IGNORECASE)
LOCAL_USER_PATH = re.compile(r"(?i)([A-Z]:[\\/]+Users[\\/]+(?!<you>)[^\\/,\s;:\"'<>]+[\\/]+[^\s,;\"'<>]*)")
PERSONAL_IDENTIFIER = re.compile(r"\b(?:DrStrangel0ve|arnav)\b", re.IGNORECASE)
PUBLIC_REPO_WORDING = re.compile(
    r"\b(?:public repo|public GitHub|checked into the repository|public project|public release)\b",
    re.IGNORECASE,
)
ACKNOWLEDGMENT_SECTION = re.compile(r"^(?:\\section\*?\{Acknowledg|##+\s+Acknowledg)", re.IGNORECASE | re.MULTILINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit paper-facing submission artifacts for blind-review/anonymity risks."
    )
    parser.add_argument("--repo-root", default=".", help="Repository root for resolving artifact paths.")
    parser.add_argument(
        "--artifacts",
        nargs="*",
        default=DEFAULT_ARTIFACTS,
        help="Paper-facing artifacts to scan.",
    )
    parser.add_argument(
        "--out-path",
        default="reports/submission_anonymity_audit_2026_06_15.md",
        help="Markdown anonymity audit report to write.",
    )
    parser.add_argument(
        "--findings-out",
        default="reports/assets/submission_anonymity_audit.csv",
        help="Machine-readable anonymity findings to write.",
    )
    parser.add_argument(
        "--checks-out",
        default="reports/assets/submission_anonymity_audit_checks.csv",
        help="Machine-readable pass/fail checks to write.",
    )
    parser.add_argument(
        "--run-date",
        default=DEFAULT_RUN_DATE.isoformat(),
        help="Date to stamp into the report, in YYYY-MM-DD format. Defaults to today's local date.",
    )
    return parser.parse_args()


def _markdown_escape(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).replace("\n", " ").replace("|", "\\|")


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame[columns].itertuples(index=False):
        lines.append("| " + " | ".join(_markdown_escape(value) for value in row) + " |")
    return "\n".join(lines)


def _snippet(text: str, start: int, end: int) -> str:
    left = max(0, start - 70)
    right = min(len(text), end + 70)
    return text[left:right].replace("\r", " ").replace("\n", " ")


def _line_number(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _display_path(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _add_match_findings(
    rows: list[dict[str, object]],
    relative_path: str,
    text: str,
    pattern: re.Pattern[str],
    issue: str,
    severity: str,
) -> None:
    for match in pattern.finditer(text):
        rows.append(
            {
                "path": relative_path,
                "issue": issue,
                "severity": severity,
                "line": _line_number(text, match.start()),
                "example": _snippet(text, match.start(), match.end()),
            }
        )


def _add_check(rows: list[dict[str, object]], check: str, passed: bool, detail: str) -> None:
    rows.append({"check": check, "passed": bool(passed), "detail": detail})


def _scan_artifact(repo_root: Path, relative_path: str) -> list[dict[str, object]]:
    full_path = repo_root / relative_path
    if not full_path.exists():
        return [
            {
                "path": relative_path,
                "issue": "missing_artifact",
                "severity": "blocker",
                "line": "",
                "example": "paper-facing artifact is missing",
            }
        ]
    text = full_path.read_text(encoding="utf-8", errors="ignore")
    rows: list[dict[str, object]] = []
    for match in AUTHOR_FIELD.finditer(text):
        author = match.group("author").strip()
        if author and "anonymous" not in author.lower():
            rows.append(
                {
                    "path": relative_path,
                    "issue": "non_anonymous_author_field",
                    "severity": "blocker",
                    "line": _line_number(text, match.start()),
                    "example": _snippet(text, match.start(), match.end()),
                }
            )
    _add_match_findings(rows, relative_path, text, EMAIL_ADDRESS, "email_address", "blocker")
    _add_match_findings(rows, relative_path, text, GITHUB_URL, "github_url", "blocker")
    _add_match_findings(rows, relative_path, text, LOCAL_USER_PATH, "local_absolute_path", "blocker")
    _add_match_findings(rows, relative_path, text, PERSONAL_IDENTIFIER, "personal_identifier", "blocker")
    _add_match_findings(rows, relative_path, text, PUBLIC_REPO_WORDING, "public_repo_wording", "review")
    _add_match_findings(rows, relative_path, text, ACKNOWLEDGMENT_SECTION, "acknowledgment_section", "review")
    return rows


def lint_anonymity(
    repo_root: Path,
    artifacts: list[str],
    out_path: Path,
    findings_out: Path,
    checks_out: Path,
    run_date: date = DEFAULT_RUN_DATE,
) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    repo_root = repo_root.resolve()
    rows: list[dict[str, object]] = []
    for artifact in artifacts:
        rows.extend(_scan_artifact(repo_root, artifact.replace("\\", "/")))

    findings = pd.DataFrame(rows, columns=["path", "issue", "severity", "line", "example"])
    blockers = findings[findings["severity"] == "blocker"]
    review = findings[findings["severity"] == "review"]

    check_rows: list[dict[str, object]] = []
    _add_check(
        check_rows,
        "paper-facing artifacts exist",
        not (findings["issue"] == "missing_artifact").any() if not findings.empty else True,
        "all configured artifacts present",
    )
    _add_check(
        check_rows,
        "author fields anonymous",
        not (findings["issue"] == "non_anonymous_author_field").any() if not findings.empty else True,
        "all LaTeX author fields are anonymous placeholders",
    )
    _add_check(
        check_rows,
        "direct identifiers absent",
        blockers.empty,
        f"{len(blockers)} blocker finding(s)",
    )
    _add_check(
        check_rows,
        "blind-review wording reviewed",
        True,
        f"{len(review)} review finding(s) for public-repo or acknowledgment wording",
    )
    checks = pd.DataFrame(check_rows)
    passed = int(checks["passed"].sum())
    total = int(len(checks))
    status = "PASS" if passed == total else "FAIL"

    findings_out = findings_out if findings_out.is_absolute() else repo_root / findings_out
    checks_out = checks_out if checks_out.is_absolute() else repo_root / checks_out
    findings_out.parent.mkdir(parents=True, exist_ok=True)
    checks_out.parent.mkdir(parents=True, exist_ok=True)
    findings.to_csv(findings_out, index=False)
    checks.to_csv(checks_out, index=False)

    summary = (
        findings.groupby(["severity", "issue"], dropna=False).size().reset_index(name="findings")
        if not findings.empty
        else pd.DataFrame(columns=["severity", "issue", "findings"])
    )
    blocker_frame = blockers.head(20)
    review_frame = review.head(20)
    lines = [
        "# Submission Anonymity Audit",
        "",
        f"Run date: {run_date.isoformat()}",
        "",
        f"Status: **{status}** ({passed}/{total} checks passed).",
        "",
        "Generated by `scripts/lint_submission_anonymity.py` from WIFS/DFF paper-facing drafts and skeletons.",
        "",
        "This audit is conservative: direct author/account/path identifiers are blockers, while generic public-repository wording is a review item because the public repo is useful for artifact release but may need anonymized phrasing in blind-review PDFs.",
        "",
        "## Checks",
        "",
        _markdown_table(checks, ["check", "passed", "detail"]),
        "",
        "## Finding Summary",
        "",
        "No findings." if summary.empty else _markdown_table(summary, ["severity", "issue", "findings"]),
        "",
        "## Blockers",
        "",
        "None." if blocker_frame.empty else _markdown_table(blocker_frame, ["path", "issue", "line", "example"]),
        "",
        "## Review Items",
        "",
        "None." if review_frame.empty else _markdown_table(review_frame, ["path", "issue", "line", "example"]),
        "",
        f"Machine-readable findings: `{_display_path(repo_root, findings_out)}`",
        "",
        f"Machine-readable checks: `{_display_path(repo_root, checks_out)}`",
        "",
    ]
    text = "\n".join(lines)
    out_path = out_path if out_path.is_absolute() else repo_root / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return text, findings, checks


def main() -> None:
    args = parse_args()
    text, _findings, checks = lint_anonymity(
        repo_root=Path(args.repo_root),
        artifacts=list(args.artifacts),
        out_path=Path(args.out_path),
        findings_out=Path(args.findings_out),
        checks_out=Path(args.checks_out),
        run_date=date.fromisoformat(args.run_date),
    )
    print(Path(args.out_path).resolve())
    print(Path(args.findings_out).resolve())
    print(Path(args.checks_out).resolve())
    if not text:
        raise SystemExit("No anonymity audit report generated")
    if not checks["passed"].all():
        raise SystemExit("Submission anonymity audit failed")


if __name__ == "__main__":
    main()
