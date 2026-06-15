from __future__ import annotations

import argparse
import hashlib
import re
from datetime import date
from pathlib import Path

import pandas as pd


DEFAULT_RUN_DATE = date.today()

TEXT_EXTENSIONS = {".bib", ".csv", ".json", ".md", ".tex", ".txt"}

DEFAULT_ARTIFACTS = [
    {
        "path": "reports/wifs_manuscript_draft_2026_06_14.md",
        "section": "paper-drafts",
        "venues": "WIFS",
        "required": True,
        "purpose": "Anonymous WIFS markdown manuscript draft.",
    },
    {
        "path": "reports/dff_manuscript_draft_2026_06_14.md",
        "section": "paper-drafts",
        "venues": "DFF",
        "required": True,
        "purpose": "Anonymous DFF markdown manuscript draft.",
    },
    {
        "path": "reports/assets/paper_skeletons/wifs_2026_paper_skeleton.tex",
        "section": "paper-drafts",
        "venues": "WIFS",
        "required": True,
        "purpose": "WIFS LaTeX skeleton with anonymous author placeholder and generated assets.",
    },
    {
        "path": "reports/assets/paper_skeletons/dff_2026_workshop_skeleton.tex",
        "section": "paper-drafts",
        "venues": "DFF",
        "required": True,
        "purpose": "DFF LaTeX skeleton with anonymous author placeholder and generated assets.",
    },
    {
        "path": "reports/paper_section_drafts_2026_06_14.md",
        "section": "writing",
        "venues": "WIFS,DFF",
        "required": True,
        "purpose": "Section prose source for paper assembly.",
    },
    {
        "path": "reports/submission_text_drafts_2026_06_14.md",
        "section": "writing",
        "venues": "DFRWS,WIFS,DFF",
        "required": True,
        "purpose": "Venue abstracts, result sentences, and paper-structure notes.",
    },
    {
        "path": "reports/submission_result_tables_2026_06_14.md",
        "section": "tables",
        "venues": "DFRWS,WIFS,DFF",
        "required": True,
        "purpose": "Copy-ready compact result tables.",
    },
    {
        "path": "reports/paired_seed_statistical_support_2026_06_15.md",
        "section": "statistical-support",
        "venues": "WIFS,DFF",
        "required": False,
        "purpose": "Paired-seed bootstrap support report for conservative model-comparison wording.",
    },
    {
        "path": "reports/assets/paired_seed_statistical_support.csv",
        "section": "statistical-support",
        "venues": "WIFS,DFF",
        "required": False,
        "purpose": "Machine-readable paired-seed deltas, bootstrap intervals, and support labels.",
    },
    {
        "path": "reports/assets/latex_tables/same_domain_anchor.tex",
        "section": "tables",
        "venues": "WIFS,DFF",
        "required": True,
        "purpose": "Same-domain anchor LaTeX table fragment.",
    },
    {
        "path": "reports/assets/latex_tables/transfer_frontier.tex",
        "section": "tables",
        "venues": "WIFS,DFF",
        "required": True,
        "purpose": "Transfer-frontier LaTeX table fragment.",
    },
    {
        "path": "reports/assets/latex_tables/robustness_stress.tex",
        "section": "tables",
        "venues": "WIFS,DFF",
        "required": True,
        "purpose": "Robustness-stress LaTeX table fragment.",
    },
    {
        "path": "reports/assets/latex_tables/reconstruction_ablation.tex",
        "section": "tables",
        "venues": "WIFS,DFF",
        "required": True,
        "purpose": "Reconstruction-ablation LaTeX table fragment.",
    },
    {
        "path": "reports/assets/latex_tables/calibration_operating_modes.tex",
        "section": "tables",
        "venues": "WIFS,DFF",
        "required": True,
        "purpose": "Calibration operating-mode LaTeX table fragment.",
    },
    {
        "path": "reports/assets/latex_tables/paired_seed_support.tex",
        "section": "tables",
        "venues": "WIFS,DFF",
        "required": True,
        "purpose": "Paired-seed statistical support LaTeX table fragment.",
    },
    {
        "path": "reports/assets/latex_tables/source_holdout_stress.tex",
        "section": "tables",
        "venues": "WIFS,DFF",
        "required": True,
        "purpose": "Held-out generator stress LaTeX table fragment.",
    },
    {
        "path": "reports/assets/claim_evidence_matrix.csv",
        "section": "claim-audit",
        "venues": "DFRWS,WIFS,DFF",
        "required": True,
        "purpose": "Machine-readable claim-to-evidence matrix.",
    },
    {
        "path": "reports/assets/claim_evidence_matrix.md",
        "section": "claim-audit",
        "venues": "DFRWS,WIFS,DFF",
        "required": True,
        "purpose": "Readable claim-to-evidence matrix.",
    },
    {
        "path": "reports/assets/source_holdout_generator_stress.png",
        "section": "figures",
        "venues": "DFRWS,WIFS,DFF",
        "required": True,
        "purpose": "Held-out generator stress figure.",
    },
    {
        "path": "reports/assets/source_holdout_generator_stress.svg",
        "section": "figures",
        "venues": "DFRWS,WIFS,DFF",
        "required": True,
        "purpose": "Editable held-out generator stress figure.",
    },
    {
        "path": "reports/assets/publication_reverse_fusion_tradeoff.png",
        "section": "figures",
        "venues": "WIFS,DFF",
        "required": False,
        "purpose": "Reverse fusion operating-point tradeoff figure.",
    },
    {
        "path": "reports/assets/publication_score_fusion_clip_frontier.png",
        "section": "figures",
        "venues": "DFRWS,DFF",
        "required": False,
        "purpose": "CLIP transfer frontier figure.",
    },
    {
        "path": "reports/assets/publication_reverse_transform_robustness.png",
        "section": "figures",
        "venues": "DFRWS,WIFS,DFF",
        "required": False,
        "purpose": "Reverse-transfer transform robustness figure.",
    },
    {
        "path": "reports/paper_section_drafts_lint_2026_06_14.md",
        "section": "quality-control",
        "venues": "WIFS,DFF",
        "required": False,
        "purpose": "Section-draft lint report.",
    },
    {
        "path": "reports/paper_skeleton_lint_2026_06_14.md",
        "section": "quality-control",
        "venues": "WIFS,DFF",
        "required": False,
        "purpose": "Paper-skeleton lint report.",
    },
    {
        "path": "reports/manuscript_drafts_lint_2026_06_14.md",
        "section": "quality-control",
        "venues": "WIFS,DFF",
        "required": False,
        "purpose": "Manuscript-draft lint report.",
    },
]

IDENTIFIER_PATTERNS = [
    ("email_address", "blocker", re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")),
    ("github_url", "blocker", re.compile(r"https?://(?:www\.)?github\.com/[^\s)>]+", re.IGNORECASE)),
    ("local_absolute_path", "blocker", re.compile(r"\b[A-Z]:\\Users\\[^\\\s]+", re.IGNORECASE)),
    ("personal_identifier", "blocker", re.compile(r"\b(?:DrStrangel0ve|arnav)\b", re.IGNORECASE)),
    (
        "public_release_wording",
        "review",
        re.compile(
            r"\b(?:public repo|public repository|public GitHub|checked into the repository|public project)\b",
            re.IGNORECASE,
        ),
    ),
    ("acknowledgment_section", "review", re.compile(r"^#+\s*acknowledg", re.IGNORECASE | re.MULTILINE)),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an anonymous-review supplementary artifact manifest for WIFS/DFF-style submissions."
    )
    parser.add_argument("--repo-root", default=".", help="Repository root for resolving artifact paths.")
    parser.add_argument(
        "--artifacts",
        nargs="*",
        default=None,
        help="Optional custom artifact paths. Defaults to the WIFS/DFF anonymous-review bundle.",
    )
    parser.add_argument(
        "--out-path",
        default="reports/anonymous_review_bundle_2026_06_15.md",
        help="Markdown bundle manifest report to write.",
    )
    parser.add_argument(
        "--manifest-out",
        default="reports/assets/anonymous_review_bundle_manifest.csv",
        help="Machine-readable anonymous bundle manifest to write.",
    )
    parser.add_argument(
        "--run-date",
        default=DEFAULT_RUN_DATE.isoformat(),
        help="Date to stamp into the report, in YYYY-MM-DD format. Defaults to today's local date.",
    )
    return parser.parse_args()


def _repo_path(repo_root: Path, path: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _repo_relative(repo_root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root).as_posix()
    except ValueError:
        return path.as_posix()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def _custom_artifacts(paths: list[str] | None) -> list[dict[str, object]]:
    if not paths:
        return DEFAULT_ARTIFACTS
    return [
        {
            "path": path.replace("\\", "/"),
            "section": "custom",
            "venues": "WIFS,DFF",
            "required": True,
            "purpose": "Custom anonymous-review artifact.",
        }
        for path in paths
    ]


def _scan_text(path: Path) -> tuple[list[str], list[str]]:
    text = path.read_text(encoding="utf-8", errors="ignore")
    blockers: list[str] = []
    review: list[str] = []
    for issue, severity, pattern in IDENTIFIER_PATTERNS:
        if pattern.search(text):
            if severity == "blocker":
                blockers.append(issue)
            else:
                review.append(issue)
    return sorted(set(blockers)), sorted(set(review))


def build_anonymous_review_bundle(
    repo_root: Path,
    artifacts: list[dict[str, object]],
    out_path: Path,
    manifest_out: Path,
    run_date: date = DEFAULT_RUN_DATE,
) -> tuple[str, pd.DataFrame]:
    repo_root = repo_root.resolve()
    out_path = _repo_path(repo_root, out_path)
    manifest_out = _repo_path(repo_root, manifest_out)
    rows = []
    for artifact in artifacts:
        relative_path = str(artifact["path"]).replace("\\", "/")
        full_path = repo_root / relative_path
        exists = full_path.exists()
        extension = full_path.suffix.lower()
        text_scanned = bool(exists and extension in TEXT_EXTENSIONS)
        blocker_issues: list[str] = []
        review_issues: list[str] = []
        if text_scanned:
            blocker_issues, review_issues = _scan_text(full_path)
        anonymous_status = (
            "missing"
            if not exists
            else "blocker"
            if blocker_issues
            else "review"
            if review_issues
            else "safe"
            if text_scanned
            else "not_scanned_binary"
        )
        rows.append(
            {
                "path": relative_path,
                "section": artifact["section"],
                "venues": artifact["venues"],
                "required": bool(artifact["required"]),
                "purpose": artifact["purpose"],
                "exists": bool(exists),
                "size_bytes": int(full_path.stat().st_size) if exists else pd.NA,
                "sha256": _sha256_file(full_path) if exists else pd.NA,
                "text_scanned": text_scanned,
                "anonymous_status": anonymous_status,
                "blocker_issues": ";".join(blocker_issues),
                "review_issues": ";".join(review_issues),
            }
        )
    manifest = pd.DataFrame(rows)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_out, index=False)

    required_missing = manifest[manifest["required"] & ~manifest["exists"]]
    identifier_findings = manifest[manifest["anonymous_status"].isin(["blocker", "review"])]
    artifact_checks = manifest.apply(
        lambda row: (bool(row["exists"]) or not bool(row["required"]))
        and row["anonymous_status"] not in {"blocker", "review"},
        axis=1,
    )
    passed = int(artifact_checks.sum())
    total = int(len(artifact_checks))
    ready = passed == total
    status = "PASS" if ready else "FAIL"
    section_summary = (
        manifest.groupby(["section", "anonymous_status"], dropna=False)
        .agg(artifacts=("path", "count"), required=("required", "sum"), bytes=("size_bytes", "sum"))
        .reset_index()
    )
    status_summary = manifest["anonymous_status"].value_counts().rename_axis("anonymous_status").reset_index(
        name="artifacts"
    )
    lines = [
        "# Anonymous Review Bundle",
        "",
        f"Run date: {run_date.isoformat()}",
        "",
        f"Status: **{status}** ({passed}/{total} checks passed).",
        "",
        "Generated by `scripts/build_anonymous_review_bundle.py` from the WIFS/DFF paper-facing drafts, tables, figures, and claim-audit assets.",
        "",
        f"This manifest is the upload checklist for an anonymous supplementary artifact package. It indexes {len(manifest)} artifacts. Text artifacts are scanned for author/account/path identifiers and release-location wording; binary figures are hashed and listed without text scanning.",
        "",
        "## Status Summary",
        "",
        _markdown_table(status_summary, ["anonymous_status", "artifacts"]),
        "",
        "## Section Summary",
        "",
        _markdown_table(section_summary, ["section", "anonymous_status", "artifacts", "required", "bytes"]),
        "",
        "## Required Missing",
        "",
        "None." if required_missing.empty else _markdown_table(required_missing, ["path", "section", "venues"]),
        "",
        "## Identifier Findings",
        "",
        "None."
        if identifier_findings.empty
        else _markdown_table(identifier_findings, ["path", "anonymous_status", "blocker_issues", "review_issues"]),
        "",
        "## Bundle Manifest",
        "",
        _markdown_table(
            manifest,
            ["path", "section", "venues", "required", "exists", "anonymous_status", "sha256"],
        ),
        "",
        f"Machine-readable manifest: `{_repo_relative(repo_root, manifest_out)}`",
        "",
    ]
    text = "\n".join(lines)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return text, manifest


def main() -> None:
    args = parse_args()
    text, manifest = build_anonymous_review_bundle(
        repo_root=Path(args.repo_root),
        artifacts=_custom_artifacts(args.artifacts),
        out_path=Path(args.out_path),
        manifest_out=Path(args.manifest_out),
        run_date=date.fromisoformat(args.run_date),
    )
    print(Path(args.out_path).resolve())
    print(Path(args.manifest_out).resolve())
    if not text:
        raise SystemExit("No anonymous review bundle report generated")
    if not manifest[manifest["required"] & ~manifest["exists"]].empty:
        raise SystemExit("Anonymous review bundle has missing required artifacts")
    if manifest["anonymous_status"].isin(["blocker", "review"]).any():
        raise SystemExit("Anonymous review bundle has identifier findings")


if __name__ == "__main__":
    main()
