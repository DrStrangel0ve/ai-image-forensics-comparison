from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import ModuleType

import pandas as pd
from PIL import Image


POSTER_PACKAGE_FILES = [
    "reports/dfrws_poster_brief_2026_06_13.md",
    "reports/assets/dfrws_poster_key_numbers.csv",
    "reports/dfrws_poster_native_figures_2026_06_13.md",
    "reports/assets/dfrws_poster_transfer_panel.csv",
    "reports/assets/dfrws_poster_transfer_panel.png",
    "reports/assets/dfrws_poster_transfer_panel.svg",
    "reports/assets/dfrws_poster_robustness_panel.csv",
    "reports/assets/dfrws_poster_robustness_panel.png",
    "reports/assets/dfrws_poster_robustness_panel.svg",
    "reports/dfrws_poster_draft_v2_2026_06_13.md",
    "reports/assets/dfrws_poster_draft_v2_2026_06_13.png",
    "reports/assets/dfrws_poster_draft_v2_2026_06_13.pptx",
]

REQUIRED_BRIEF_SECTIONS = [
    "## Working Title",
    "## One-Sentence Claim",
    "## Poster Spine",
    "## Key Numbers",
    "## Robustness Stress Panel",
    "## Figure Package",
    "## Claims To Carry",
    "## Do Not Overclaim",
    "## Immediate Poster TODOs",
]

REQUIRED_BRIEF_CAUTIONS = [
    "Do not claim SCP-Fusion beats frozen CLIP",
    "single-image physical/signal proxy",
    "JPEG30, blur, resize, and screenshot transforms still expose weaknesses",
]

PANEL_FILES = {
    "transfer": "reports/assets/dfrws_poster_transfer_panel.png",
    "robustness": "reports/assets/dfrws_poster_robustness_panel.png",
}

POSTER_PREVIEW = "reports/assets/dfrws_poster_draft_v2_2026_06_13.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lint DFRWS poster assets against canonical results and poster-ready asset checks."
    )
    parser.add_argument("--repo-root", default=".", help="Repository root for resolving generated assets.")
    parser.add_argument(
        "--core-results",
        default="reports/assets/publication_core_results.csv",
        help="Canonical publication result table.",
    )
    parser.add_argument(
        "--claim-matrix",
        default="reports/assets/claim_evidence_matrix.csv",
        help="Claim/evidence matrix used by the poster brief builder.",
    )
    parser.add_argument(
        "--brief",
        default="reports/dfrws_poster_brief_2026_06_13.md",
        help="Generated DFRWS poster brief.",
    )
    parser.add_argument(
        "--key-numbers",
        default="reports/assets/dfrws_poster_key_numbers.csv",
        help="Generated DFRWS key-number CSV.",
    )
    parser.add_argument(
        "--transfer-panel",
        default="reports/assets/dfrws_poster_transfer_panel.csv",
        help="Generated transfer-panel source CSV.",
    )
    parser.add_argument(
        "--robustness-panel",
        default="reports/assets/dfrws_poster_robustness_panel.csv",
        help="Generated robustness-panel source CSV.",
    )
    parser.add_argument("--min-panel-width", type=int, default=2500)
    parser.add_argument("--min-panel-height", type=int, default=1000)
    parser.add_argument("--min-preview-width", type=int, default=1000)
    parser.add_argument("--min-preview-height", type=int, default=600)
    parser.add_argument(
        "--out-path",
        default="reports/dfrws_poster_package_lint_2026_06_14.md",
        help="Markdown lint report to write.",
    )
    parser.add_argument(
        "--checks-out",
        default="reports/assets/dfrws_poster_package_lint.csv",
        help="Machine-readable lint checks to write.",
    )
    return parser.parse_args()


def _add_check(rows: list[dict[str, object]], check: str, passed: bool, detail: str) -> None:
    rows.append({"check": check, "passed": bool(passed), "detail": detail})


def _load_script(repo_root: Path, filename: str, module_name: str) -> ModuleType:
    script_path = repo_root / "scripts" / filename
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _values_match(left: object, right: object, tolerance: float = 1e-9) -> bool:
    if pd.isna(left) and pd.isna(right):
        return True
    if pd.isna(left) or pd.isna(right):
        return False
    if isinstance(left, str) or isinstance(right, str):
        return str(left) == str(right)
    return abs(float(left) - float(right)) <= tolerance


def _compare_frames(left: pd.DataFrame, right: pd.DataFrame, columns: list[str]) -> list[str]:
    mismatches: list[str] = []
    if len(left) != len(right):
        return [f"row_count:{len(left)}!={len(right)}"]
    for index, (left_row, right_row) in enumerate(zip(left.itertuples(index=False), right.itertuples(index=False))):
        for column in columns:
            left_value = getattr(left_row, column)
            right_value = getattr(right_row, column)
            if not _values_match(left_value, right_value):
                mismatches.append(f"row{index}:{column}")
    return mismatches


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def lint_dfrws_poster_package(
    repo_root: Path,
    core_results_path: Path,
    claim_matrix_path: Path,
    brief_path: Path,
    key_numbers_path: Path,
    transfer_panel_path: Path,
    robustness_panel_path: Path,
    min_panel_width: int,
    min_panel_height: int,
    min_preview_width: int,
    min_preview_height: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    repo_root = repo_root.resolve()
    brief_builder = _load_script(repo_root, "build_dfrws_poster_brief.py", "dfrws_poster_brief_builder")
    figure_builder = _load_script(repo_root, "build_dfrws_poster_figures.py", "dfrws_poster_figure_builder")

    core = pd.read_csv(core_results_path)
    brief_text = brief_path.read_text(encoding="utf-8")
    key_numbers = pd.read_csv(key_numbers_path)
    transfer_panel = pd.read_csv(transfer_panel_path)
    robustness_panel = pd.read_csv(robustness_panel_path)

    missing_files = [relative for relative in POSTER_PACKAGE_FILES if not (repo_root / relative).exists()]
    _add_check(
        rows,
        "DFRWS poster package files exist",
        not missing_files,
        "all poster package files are present" if not missing_files else ", ".join(missing_files),
    )

    figure_missing = [
        file_path
        for _panel, file_path, _poster_use in brief_builder.FIGURE_PACKAGE
        if not (repo_root / file_path).exists()
    ]
    _add_check(
        rows,
        "brief figure package files exist",
        not figure_missing,
        "all brief figure files are present" if not figure_missing else ", ".join(figure_missing),
    )

    for section in REQUIRED_BRIEF_SECTIONS:
        _add_check(
            rows,
            f"brief section present: {section}",
            section in brief_text,
            "present" if section in brief_text else "missing",
        )

    for caution in REQUIRED_BRIEF_CAUTIONS:
        _add_check(
            rows,
            f"brief caution present: {caution}",
            caution in brief_text,
            "present" if caution in brief_text else "missing",
        )

    _expected_text, expected_key_numbers = brief_builder.build_poster_brief(
        Path(core_results_path),
        Path(claim_matrix_path),
    )
    key_columns = ["finding", "method", "setting", "metrics", "poster_use"]
    key_mismatches = _compare_frames(key_numbers[key_columns], expected_key_numbers[key_columns], key_columns)
    _add_check(
        rows,
        "poster key numbers match canonical results",
        not key_mismatches,
        "key-number CSV matches poster brief builder" if not key_mismatches else ", ".join(key_mismatches),
    )

    expected_transfer = pd.concat(
        [
            figure_builder._panel_frame(core, figure_builder.TRANSFER_SAME_DOMAIN, "Ishu same-domain"),
            figure_builder._panel_frame(core, figure_builder.TRANSFER_CROSS_DOMAIN, "Ishu -> MS COCOAI"),
            figure_builder._panel_frame(
                core,
                [(figure_builder.TRIAGE_CALLOUT, "Frozen CLIP triage")],
                "5% source-heldout triage",
            ),
        ],
        ignore_index=True,
    )
    transfer_columns = list(expected_transfer.columns)
    transfer_mismatches = _compare_frames(transfer_panel[transfer_columns], expected_transfer, transfer_columns)
    _add_check(
        rows,
        "transfer panel CSV matches canonical results",
        not transfer_mismatches,
        "transfer panel CSV matches figure builder" if not transfer_mismatches else ", ".join(transfer_mismatches),
    )

    expected_robustness = figure_builder._panel_frame(
        core,
        figure_builder.ROBUSTNESS_ROWS,
        "MS COCOAI -> Ishu tuned fusion robustness",
    )
    robustness_columns = list(expected_robustness.columns)
    robustness_mismatches = _compare_frames(
        robustness_panel[robustness_columns],
        expected_robustness,
        robustness_columns,
    )
    _add_check(
        rows,
        "robustness panel CSV matches canonical results",
        not robustness_mismatches,
        "robustness panel CSV matches figure builder" if not robustness_mismatches else ", ".join(robustness_mismatches),
    )

    for panel, relative in PANEL_FILES.items():
        path = repo_root / relative
        if not path.exists():
            _add_check(rows, f"{panel} panel image dimensions", False, f"{relative} missing")
            continue
        width, height = _image_size(path)
        _add_check(
            rows,
            f"{panel} panel image dimensions",
            width >= min_panel_width and height >= min_panel_height,
            f"{relative}: {width}x{height}; minimum {min_panel_width}x{min_panel_height}",
        )

    preview_path = repo_root / POSTER_PREVIEW
    if preview_path.exists():
        width, height = _image_size(preview_path)
        preview_ok = width >= min_preview_width and height >= min_preview_height
        detail = f"{POSTER_PREVIEW}: {width}x{height}; minimum {min_preview_width}x{min_preview_height}"
    else:
        preview_ok = False
        detail = f"{POSTER_PREVIEW} missing"
    _add_check(rows, "poster draft preview dimensions", preview_ok, detail)

    pptx_path = repo_root / "reports/assets/dfrws_poster_draft_v2_2026_06_13.pptx"
    pptx_size = pptx_path.stat().st_size if pptx_path.exists() else 0
    _add_check(
        rows,
        "poster draft PPTX is nontrivial",
        pptx_size > 10000,
        f"size_bytes={pptx_size}; minimum 10000",
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
        "# DFRWS Poster Package Lint",
        "",
        "Run date: 2026-06-14",
        "",
        f"Status: **{status}** ({passed}/{total} checks passed).",
        "",
        "Generated by `scripts/lint_dfrws_poster_package.py` from the DFRWS poster brief, poster panels, and canonical result table.",
        "",
        _markdown_table(frame),
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    frame = lint_dfrws_poster_package(
        repo_root=Path(args.repo_root),
        core_results_path=Path(args.core_results),
        claim_matrix_path=Path(args.claim_matrix),
        brief_path=Path(args.brief),
        key_numbers_path=Path(args.key_numbers),
        transfer_panel_path=Path(args.transfer_panel),
        robustness_panel_path=Path(args.robustness_panel),
        min_panel_width=args.min_panel_width,
        min_panel_height=args.min_panel_height,
        min_preview_width=args.min_preview_width,
        min_preview_height=args.min_preview_height,
    )
    checks_path = Path(args.checks_out)
    checks_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(checks_path, index=False)
    write_report(frame, Path(args.out_path))
    if not bool(frame["passed"].all()):
        failed = frame[~frame["passed"]]
        raise SystemExit(
            "DFRWS poster package lint failed: "
            + "; ".join(f"{row.check} ({row.detail})" for row in failed.itertuples(index=False))
        )
    print(Path(args.out_path).resolve())
    print(checks_path.resolve())


if __name__ == "__main__":
    main()
