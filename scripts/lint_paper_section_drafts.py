from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


REQUIRED_SECTIONS = {
    "WIFS Introduction Draft": {
        "word_range": (100, 260),
        "phrases": ["source-heldout", "ranking", "calibration", "paired-seed support", "0.8450", "0.8641"],
    },
    "WIFS Related Work Draft": {
        "word_range": (80, 180),
        "phrases": ["universal_fake_detectors_2023", "single-image physical/signal proxy", "photometric stereo"],
    },
    "WIFS Data And Audit Draft": {
        "word_range": (80, 180),
        "phrases": ["Ishu AI-vs-real", "MS COCOAI", "raw datasets", "artifact manifests"],
    },
    "WIFS Methods Draft": {
        "word_range": (80, 180),
        "phrases": ["combined_v3", "ResNet-18", "SCP-Fusion", "Brier score", "ECE"],
    },
    "WIFS Results Draft": {
        "word_range": (120, 360),
        "phrases": [
            "0.8246",
            "0.8450",
            "0.8641",
            "0.7749",
            "paired-seed support",
            "JPEG30",
            "blur",
            "reconstruction_v2",
        ],
    },
    "DFF Expansion Draft": {
        "word_range": (100, 280),
        "phrases": [
            "diagnostic protocol",
            "Paired-seed support",
            "combined_v4",
            "AEROBLADE/FIRE-style",
            "ablation candidate",
            "reconstruction_v2",
        ],
    },
    "Limitations And Reproducibility Draft": {
        "word_range": (80, 180),
        "phrases": [
            "single-image proxy",
            "does not universally beat frozen CLIP",
            "official venue templates",
            "verified bibliography metadata",
        ],
    },
}

FORBIDDEN_PATTERNS = [
    r"\bstate[- ]of[- ]the[- ]art\b",
    r"\bproduction[- ]ready\b",
    r"\bguaranteed\b",
    r"\bSCP-Fusion beats CLIP\b",
    r"\buniversally beats\b",
    r"\brobust to processing\b",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lint generated WIFS/DFF paper section prose for required evidence, caveats, and overclaims."
    )
    parser.add_argument("--section-drafts", default="reports/paper_section_drafts_2026_06_14.md")
    parser.add_argument("--manifest", default="reports/assets/paper_section_draft_manifest.csv")
    parser.add_argument("--out-path", default="reports/paper_section_drafts_lint_2026_06_14.md")
    parser.add_argument("--checks-out", default="reports/assets/paper_section_drafts_lint.csv")
    return parser.parse_args()


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))


def _extract_section(text: str, header: str) -> str:
    pattern = rf"^## {re.escape(header)}\s*$"
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        return ""
    start = match.end()
    next_match = re.search(r"^## ", text[start:], flags=re.MULTILINE)
    end = start + next_match.start() if next_match else len(text)
    return text[start:end].strip()


def _add_check(rows: list[dict[str, object]], section: str, check: str, passed: bool, detail: str) -> None:
    rows.append({"section": section, "check": check, "passed": bool(passed), "detail": detail})


def lint_paper_section_drafts(section_drafts: Path, manifest_path: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    text = section_drafts.read_text(encoding="utf-8")
    manifest = pd.read_csv(manifest_path)
    manifest_by_section = {str(row.section): row for row in manifest.itertuples(index=False)}

    _add_check(
        rows,
        "manifest",
        "all required sections listed",
        set(REQUIRED_SECTIONS).issubset(manifest_by_section),
        "all required sections listed"
        if set(REQUIRED_SECTIONS).issubset(manifest_by_section)
        else ", ".join(sorted(set(REQUIRED_SECTIONS) - set(manifest_by_section))),
    )

    for section, requirements in REQUIRED_SECTIONS.items():
        body = _extract_section(text, section)
        _add_check(rows, section, "section exists", bool(body), "present" if body else "missing")
        if not body:
            continue

        count = _word_count(body)
        minimum, maximum = requirements["word_range"]
        recorded_count = getattr(manifest_by_section.get(section), "word_count", -1)
        _add_check(
            rows,
            section,
            "word count in range",
            minimum <= count <= maximum,
            f"{count} words; allowed {minimum}-{maximum}",
        )
        _add_check(
            rows,
            section,
            "recorded word count matches text",
            int(recorded_count) == count,
            f"computed {count}, recorded {recorded_count}",
        )
        for phrase in requirements["phrases"]:
            _add_check(
                rows,
                section,
                f"required phrase present: {phrase}",
                phrase in body,
                "present" if phrase in body else "missing",
            )

    for pattern in FORBIDDEN_PATTERNS:
        found = re.search(pattern, text, flags=re.IGNORECASE) is not None
        _add_check(
            rows,
            "global",
            f"avoid overclaim: {pattern}",
            not found,
            "not found" if not found else "found in section drafts",
        )

    return pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("\n", " ").replace("|", "\\|") for value in row) + " |")
    return "\n".join(lines)


def write_report(frame: pd.DataFrame, out_path: Path) -> None:
    passed = int(frame["passed"].sum())
    total = len(frame)
    status = "PASS" if passed == total else "FAIL"
    lines = [
        "# Paper Section Drafts Lint",
        "",
        "Run date: 2026-06-14",
        "",
        f"Status: **{status}** ({passed}/{total} checks passed).",
        "",
        "Generated by `scripts/lint_paper_section_drafts.py` from the WIFS/DFF paper section drafts and manifest.",
        "",
        _markdown_table(frame),
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    frame = lint_paper_section_drafts(Path(args.section_drafts), Path(args.manifest))
    checks_out = Path(args.checks_out)
    checks_out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(checks_out, index=False)
    write_report(frame, Path(args.out_path))
    if not bool(frame["passed"].all()):
        failed = frame[~frame["passed"]]
        raise SystemExit(
            "Paper section draft lint failed: "
            + "; ".join(f"{row.section}:{row.check} ({row.detail})" for row in failed.itertuples(index=False))
        )
    print(Path(args.out_path).resolve())
    print(checks_out.resolve())


if __name__ == "__main__":
    main()
