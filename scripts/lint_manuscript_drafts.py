from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


RUN_DATE = "2026-06-14"

REQUIRED_SECTIONS = {
    "WIFS": [
        "Abstract",
        "Introduction",
        "Related work and problem framing",
        "Data, audits, and metrics",
        "Methods",
        "Results",
        "Limitations and reproducibility",
        "Author Checklist",
    ],
    "DFF": [
        "Abstract",
        "Motivation and related work",
        "Protocol and branches",
        "Main results and operating points",
        "Failure analysis and ablations",
        "Limitations, ethics, and reproducibility",
        "Author Checklist",
    ],
}

REQUIRED_PHRASES = {
    "WIFS": [
        "0.8450 accuracy",
        "0.8641 AUC",
        "single-image physical/signal proxy",
        "SCP-Fusion does not universally beat frozen CLIP",
        "Assets to place or cite",
    ],
    "DFF": [
        "SCP-Fusion as a diagnostic protocol",
        "0.8641 AUC",
        "Failure grids and source-slice diagnostics",
        "single-image physical/signal proxy",
        "Assets to place or cite",
    ],
}

WORD_LIMITS = {
    "WIFS": (900, 2200),
    "DFF": (1000, 2600),
}

FORBIDDEN_PATTERNS = [
    r"\bstate[- ]of[- ]the[- ]art\b",
    r"\bproduction[- ]ready\b",
    r"\bguaranteed\b",
    r"\bSCP-Fusion beats CLIP\b",
    r"\brobust to processing\b",
]

ASSET_LINE_PATTERN = re.compile(
    r"^-\s+(?:Figure candidate|Table fragment|Supporting artifact):\s+`([^`]+)`\s*$",
    flags=re.MULTILINE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lint assembled WIFS/DFF manuscript drafts for structure, assets, and overclaim guardrails."
    )
    parser.add_argument("--repo-root", default=".", help="Repository root for resolving asset references.")
    parser.add_argument("--manifest", default="reports/assets/manuscript_draft_manifest.csv")
    parser.add_argument("--wifs-draft", default="reports/wifs_manuscript_draft_2026_06_14.md")
    parser.add_argument("--dff-draft", default="reports/dff_manuscript_draft_2026_06_14.md")
    parser.add_argument("--out-path", default="reports/manuscript_drafts_lint_2026_06_14.md")
    parser.add_argument("--checks-out", default="reports/assets/manuscript_drafts_lint.csv")
    return parser.parse_args()


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w.-]+\b", text))


def _extract_heading_names(text: str) -> set[str]:
    return {match.group(1).strip() for match in re.finditer(r"^##\s+(.+?)\s*$", text, flags=re.MULTILINE)}


def _add_check(rows: list[dict[str, object]], venue: str, check: str, passed: bool, detail: str) -> None:
    rows.append({"venue": venue, "check": check, "passed": bool(passed), "detail": detail})


def _draft_specs(wifs_draft: Path, dff_draft: Path) -> dict[str, Path]:
    return {"WIFS": wifs_draft, "DFF": dff_draft}


def lint_manuscript_drafts(
    repo_root: Path,
    manifest_path: Path,
    wifs_draft: Path,
    dff_draft: Path,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    repo_root = repo_root.resolve()
    manifest = pd.read_csv(manifest_path)
    manifest_by_venue = {str(row.venue): row for row in manifest.itertuples(index=False)}

    for venue, draft_path in _draft_specs(wifs_draft, dff_draft).items():
        full_draft_path = repo_root / draft_path
        exists = full_draft_path.exists()
        _add_check(rows, venue, "draft exists", exists, str(draft_path) if exists else "missing")
        if not exists:
            continue

        text = full_draft_path.read_text(encoding="utf-8")
        headings = _extract_heading_names(text)
        missing_sections = [section for section in REQUIRED_SECTIONS[venue] if section not in headings]
        _add_check(
            rows,
            venue,
            "required sections present",
            not missing_sections,
            "all required sections present" if not missing_sections else ", ".join(missing_sections),
        )

        count = _word_count(text)
        minimum, maximum = WORD_LIMITS[venue]
        _add_check(
            rows,
            venue,
            "draft word count in editable range",
            minimum <= count <= maximum,
            f"{count} words; allowed {minimum}-{maximum}",
        )
        manifest_count = int(getattr(manifest_by_venue[venue], "draft_word_count", -1))
        _add_check(
            rows,
            venue,
            "manifest word count matches draft",
            manifest_count == count,
            f"computed {count}, recorded {manifest_count}",
        )

        asset_paths = ASSET_LINE_PATTERN.findall(text)
        missing_assets = [asset for asset in asset_paths if not (repo_root / asset).exists()]
        manifest_asset_count = int(getattr(manifest_by_venue[venue], "asset_callouts", -1))
        _add_check(
            rows,
            venue,
            "asset callout count matches manifest",
            manifest_asset_count == len(asset_paths),
            f"computed {len(asset_paths)}, recorded {manifest_asset_count}",
        )
        _add_check(
            rows,
            venue,
            "asset callouts exist",
            not missing_assets,
            "all asset callouts exist" if not missing_assets else ", ".join(missing_assets),
        )

        for phrase in REQUIRED_PHRASES[venue]:
            _add_check(
                rows,
                venue,
                f"required phrase present: {phrase}",
                phrase in text,
                "present" if phrase in text else "missing",
            )

        for pattern in FORBIDDEN_PATTERNS:
            found = re.search(pattern, text, flags=re.IGNORECASE) is not None
            _add_check(
                rows,
                venue,
                f"avoid overclaim: {pattern}",
                not found,
                "not found" if not found else "found in draft",
            )

    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False):
        lines.append(
            "| "
            + " | ".join(str(value).replace("\n", " ").replace("|", "\\|") for value in row)
            + " |"
        )
    return "\n".join(lines)


def write_report(frame: pd.DataFrame, out_path: Path) -> None:
    passed = int(frame["passed"].sum())
    total = len(frame)
    status = "PASS" if passed == total else "FAIL"
    lines = [
        "# Manuscript Drafts Lint",
        "",
        f"Run date: {RUN_DATE}",
        "",
        f"Status: **{status}** ({passed}/{total} checks passed).",
        "",
        "Generated by `scripts/lint_manuscript_drafts.py` from the assembled WIFS/DFF manuscript drafts and manifest.",
        "",
        _markdown_table(frame),
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    frame = lint_manuscript_drafts(
        Path(args.repo_root),
        Path(args.manifest),
        Path(args.wifs_draft),
        Path(args.dff_draft),
    )
    checks_out = Path(args.checks_out)
    checks_out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(checks_out, index=False)
    write_report(frame, Path(args.out_path))
    if not bool(frame["passed"].all()):
        failed = frame[~frame["passed"]]
        raise SystemExit(
            "Manuscript draft lint failed: "
            + "; ".join(f"{row.venue}:{row.check} ({row.detail})" for row in failed.itertuples(index=False))
        )
    print(Path(args.out_path).resolve())
    print(checks_out.resolve())


if __name__ == "__main__":
    main()
