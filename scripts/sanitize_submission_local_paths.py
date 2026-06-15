from __future__ import annotations

import argparse
import subprocess
from datetime import date
from pathlib import Path

import pandas as pd


DEFAULT_RUN_DATE = date.today()
TEXT_SUFFIXES = {
    ".bib",
    ".cff",
    ".csv",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".tex",
    ".txt",
    ".yaml",
    ".yml",
}
SKIP_PARTS = {".git", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".venv", "__pycache__", "data", "models"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace workstation-specific repository prefixes in checked-in text artifacts with <repo>."
    )
    parser.add_argument("--repo-root", default=".", help="Repository root to sanitize.")
    parser.add_argument(
        "--manifest",
        default="reports/assets/submission_packet_manifest.csv",
        help="Optional packet manifest to include in the candidate set.",
    )
    parser.add_argument(
        "--out-path",
        default="reports/submission_path_sanitization_2026_06_15.md",
        help="Markdown sanitization report to write.",
    )
    parser.add_argument(
        "--csv-out",
        default="reports/assets/submission_path_sanitization.csv",
        help="Machine-readable sanitization report to write.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Rewrite candidate files in place. Without this flag, only report what would change.",
    )
    parser.add_argument(
        "--run-date",
        default=DEFAULT_RUN_DATE.isoformat(),
        help="Date to stamp into the report, in YYYY-MM-DD format. Defaults to today's local date.",
    )
    return parser.parse_args()


def _resolve_repo_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _repo_relative(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


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


def _path_variants(repo_root: Path) -> list[str]:
    resolved = repo_root.resolve()
    variants = {
        str(resolved),
        resolved.as_posix(),
        str(resolved).replace("/", "\\"),
        str(resolved).replace("\\", "/"),
    }
    return sorted(variants, key=len, reverse=True)


def _tracked_files(repo_root: Path) -> set[str]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return set()
    return {line.strip().replace("\\", "/") for line in result.stdout.splitlines() if line.strip()}


def _manifest_files(repo_root: Path, manifest_path: Path) -> set[str]:
    manifest_path = _resolve_repo_path(repo_root, manifest_path)
    if not manifest_path.exists():
        return set()
    manifest = pd.read_csv(manifest_path)
    return {str(path).replace("\\", "/") for path in manifest["path"].dropna()}


def _is_text_candidate(relative_path: str) -> bool:
    path = Path(relative_path)
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return False
    return not any(part in SKIP_PARTS for part in path.parts)


def _is_placeholder_summary_candidate(relative_path: str) -> bool:
    path = Path(relative_path)
    normalized = relative_path.replace("\\", "/")
    return normalized.startswith("reports/assets/") and "detail" in path.name and path.suffix.lower() == ".csv"


def _candidate_files(repo_root: Path, manifest_path: Path) -> list[str]:
    candidates = _tracked_files(repo_root) | _manifest_files(repo_root, manifest_path)
    return sorted(path for path in candidates if _is_text_candidate(path) and (repo_root / path).exists())


def _sanitize_text(text: str, variants: list[str]) -> tuple[str, int]:
    replacements = 0
    sanitized = text
    for variant in variants:
        count = sanitized.count(variant)
        if count:
            sanitized = sanitized.replace(variant, "<repo>")
            replacements += count
    return sanitized, replacements


def sanitize_local_paths(
    repo_root: Path,
    manifest_path: Path,
    out_path: Path,
    csv_out: Path,
    apply_changes: bool,
    run_date: date = DEFAULT_RUN_DATE,
) -> tuple[str, pd.DataFrame]:
    repo_root = repo_root.resolve()
    out_path = _resolve_repo_path(repo_root, out_path)
    csv_out = _resolve_repo_path(repo_root, csv_out)
    output_relpaths = {
        _repo_relative(repo_root, out_path),
        _repo_relative(repo_root, csv_out),
    }
    variants = _path_variants(repo_root)
    rows = []

    for relative_path in _candidate_files(repo_root, manifest_path):
        if relative_path in output_relpaths:
            continue
        full_path = repo_root / relative_path
        try:
            text = full_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        sanitized, replacements = _sanitize_text(text, variants)
        sanitized_placeholders = sanitized.count("<repo>")
        if replacements and apply_changes:
            full_path.write_text(sanitized, encoding="utf-8")
        if replacements or (sanitized_placeholders and _is_placeholder_summary_candidate(relative_path)):
            rows.append(
                {
                    "path": relative_path,
                    "replacements": replacements,
                    "sanitized_placeholders": sanitized_placeholders,
                    "applied": bool(apply_changes),
                }
            )

    frame = pd.DataFrame(rows, columns=["path", "replacements", "sanitized_placeholders", "applied"])
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_out, index=False)

    total_replacements = int(frame["replacements"].sum()) if not frame.empty else 0
    changed_files = int((frame["replacements"] > 0).sum()) if not frame.empty else 0
    placeholder_files = int((frame["sanitized_placeholders"] > 0).sum()) if not frame.empty else 0
    status = "APPLIED" if apply_changes else "DRY-RUN"
    preview = (
        frame.sort_values(["replacements", "sanitized_placeholders", "path"], ascending=[False, False, True]).head(20)
        if not frame.empty
        else frame
    )
    lines = [
        "# Submission Path Sanitization",
        "",
        f"Run date: {run_date.isoformat()}",
        "",
        f"Status: **{status}**.",
        "",
        "Generated by `scripts/sanitize_submission_local_paths.py` from checked-in text files plus the submission packet manifest.",
        "",
        "This utility replaces the local repository prefix with `<repo>` while leaving dataset-relative suffixes and all numeric results intact.",
        "",
        f"Files with local-prefix replacements: `{changed_files}`; replacements: `{total_replacements}`; files carrying `<repo>` placeholders: `{placeholder_files}`.",
        "",
        "## Sanitization Findings",
        "",
        "None." if preview.empty else _markdown_table(preview, ["path", "replacements", "sanitized_placeholders", "applied"]),
        "",
        f"Machine-readable report: `{csv_out.relative_to(repo_root).as_posix()}`",
        "",
    ]
    text = "\n".join(lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return text, frame


def main() -> None:
    args = parse_args()
    text, _frame = sanitize_local_paths(
        repo_root=Path(args.repo_root),
        manifest_path=Path(args.manifest),
        out_path=Path(args.out_path),
        csv_out=Path(args.csv_out),
        apply_changes=bool(args.apply),
        run_date=date.fromisoformat(args.run_date),
    )
    print(Path(args.out_path).resolve())
    print(Path(args.csv_out).resolve())
    if not text:
        raise SystemExit("No path sanitization report generated")


if __name__ == "__main__":
    main()
