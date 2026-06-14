from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RUN_DATE = "2026-06-14"

UPLOAD_ITEMS = [
    {
        "venue_key": "DFRWS",
        "venue": "DFRWS-USA 2026 poster/demo",
        "deadline": "2026-07-07",
        "item": "poster abstract",
        "status": "ready_asset",
        "paths": "reports/submission_text_drafts_2026_06_14.md; reports/assets/submission_text_drafts_word_counts.csv",
        "action": "Use the DFRWS poster abstract; current generated count is 183 words.",
    },
    {
        "venue_key": "DFRWS",
        "venue": "DFRWS-USA 2026 poster/demo",
        "deadline": "2026-07-07",
        "item": "editable poster draft",
        "status": "ready_asset",
        "paths": "reports/assets/dfrws_poster_draft_v2_2026_06_13.pptx; reports/assets/dfrws_poster_draft_v2_2026_06_13.png",
        "action": "Open the PPTX, make final visual edits, then export the upload PDF/image.",
    },
    {
        "venue_key": "DFRWS",
        "venue": "DFRWS-USA 2026 poster/demo",
        "deadline": "2026-07-07",
        "item": "poster-native panels",
        "status": "ready_asset",
        "paths": "reports/assets/dfrws_poster_transfer_panel.png; reports/assets/dfrws_poster_robustness_panel.png; reports/assets/dfrws_poster_transfer_panel.svg; reports/assets/dfrws_poster_robustness_panel.svg",
        "action": "Use the PNGs directly or edit the SVGs if final poster layout needs native vector text.",
    },
    {
        "venue_key": "DFRWS",
        "venue": "DFRWS-USA 2026 poster/demo",
        "deadline": "2026-07-07",
        "item": "qualitative grid choice",
        "status": "decision_needed",
        "paths": "reports/assets/qualitative_seed17_scp_fusion_false_negatives.png; reports/assets/qualitative_seed29_scp_fusion_false_negatives.png",
        "action": "Choose one false-negative grid for the poster; keep the companion grid for paper appendix material.",
    },
    {
        "venue_key": "DFRWS",
        "venue": "DFRWS-USA 2026 poster/demo",
        "deadline": "2026-07-07",
        "item": "public repo and citation",
        "status": "ready_asset",
        "paths": "README.md; CITATION.cff; reports/reproducibility_checklist_2026_06_12.md",
        "action": "Include the public GitHub link and cite the reproducibility checklist if the submission form allows supplemental links.",
    },
    {
        "venue_key": "DFRWS",
        "venue": "DFRWS-USA 2026 poster/demo",
        "deadline": "2026-07-07",
        "item": "final upload export",
        "status": "final_export_needed",
        "paths": "reports/assets/dfrws_poster_draft_v2_2026_06_13.pptx",
        "action": "Export the final polished poster to the exact DFRWS-required format before upload.",
    },
    {
        "venue_key": "WIFS",
        "venue": "IEEE WIFS 2026 paper",
        "deadline": "2026-07-15",
        "item": "paper skeleton",
        "status": "ready_asset",
        "paths": "reports/assets/paper_skeletons/wifs_2026_paper_skeleton.tex; reports/submission_paper_skeletons_2026_06_14.md",
        "action": "Use the IEEE skeleton as the working draft, then compress to the 6-page venue limit.",
    },
    {
        "venue_key": "WIFS",
        "venue": "IEEE WIFS 2026 paper",
        "deadline": "2026-07-15",
        "item": "paper prose sections",
        "status": "writing_needed",
        "paths": "reports/paper_section_drafts_2026_06_14.md; reports/paper_section_drafts_lint_2026_06_14.md; reports/method_family_comparison_2026_06_14.md",
        "action": "Merge generated sections into a compact WIFS draft; keep caveats about CLIP, single-image physics, and native tiling.",
    },
    {
        "venue_key": "WIFS",
        "venue": "IEEE WIFS 2026 paper",
        "deadline": "2026-07-15",
        "item": "LaTeX tables and bibliography",
        "status": "ready_asset",
        "paths": "reports/assets/latex_tables/same_domain_anchor.tex; reports/assets/latex_tables/transfer_frontier.tex; reports/assets/latex_tables/reverse_operating_points.tex; reports/assets/latex_tables/robustness_stress.tex; reports/assets/latex_tables/source_holdout_stress.tex; references.bib",
        "action": "Pull compact tables and references into the paper template; verify final citation metadata before submission.",
    },
    {
        "venue_key": "WIFS",
        "venue": "IEEE WIFS 2026 paper",
        "deadline": "2026-07-15",
        "item": "figures",
        "status": "ready_asset",
        "paths": "reports/assets/publication_score_fusion_clip_frontier.png; reports/assets/source_holdout_generator_stress.png; reports/assets/publication_triage_operating_points.png; reports/assets/publication_reverse_operating_points.png; reports/assets/publication_reverse_transform_robustness.png",
        "action": "Use a compact figure set; avoid overcrowding the 6-page paper.",
    },
    {
        "venue_key": "WIFS",
        "venue": "IEEE WIFS 2026 paper",
        "deadline": "2026-07-15",
        "item": "paper-critical breadth choice",
        "status": "decision_needed",
        "paths": "reports/submission_scorecard_2026_06_14.md; reports/submission_plan_2026.md",
        "action": "Choose one additional breadth check only if time allows: larger source split, tiled neural/foundation branch, or source-aware v4 selection.",
    },
    {
        "venue_key": "WIFS",
        "venue": "IEEE WIFS 2026 paper",
        "deadline": "2026-07-15",
        "item": "final camera-ready formatting",
        "status": "final_export_needed",
        "paths": "reports/assets/paper_skeletons/wifs_2026_paper_skeleton.tex",
        "action": "Compile, inspect page count, check references/figures, and export the final PDF for the WIFS system.",
    },
    {
        "venue_key": "DFF",
        "venue": "DFF-2026 ACM Multimedia workshop",
        "deadline": "2026-07-16",
        "item": "paper skeleton",
        "status": "ready_asset",
        "paths": "reports/assets/paper_skeletons/dff_2026_workshop_skeleton.tex; reports/submission_paper_skeletons_2026_06_14.md",
        "action": "Use the ACM-style skeleton as the working DFF draft.",
    },
    {
        "venue_key": "DFF",
        "venue": "DFF-2026 ACM Multimedia workshop",
        "deadline": "2026-07-16",
        "item": "SCP-Fusion diagnostic framing",
        "status": "writing_needed",
        "paths": "reports/paper_section_drafts_2026_06_14.md; reports/submission_text_drafts_2026_06_14.md; reports/method_family_comparison_2026_06_14.md",
        "action": "Keep SCP-Fusion framed as a diagnostic protocol, not as a universal replacement for frozen CLIP.",
    },
    {
        "venue_key": "DFF",
        "venue": "DFF-2026 ACM Multimedia workshop",
        "deadline": "2026-07-16",
        "item": "failure-analysis material",
        "status": "ready_asset",
        "paths": "reports/assets/qualitative_seed17_scp_fusion_false_negatives.png; reports/assets/qualitative_seed29_scp_fusion_false_negatives.png; reports/combined_v4_source_slice_diagnostics_2026_06_13.md; reports/source_holdout_generator_stress_2026_06_14.md; reports/assets/source_holdout_generator_stress.png; reports/assets/submission_table_source_holdout_stress.csv; reports/assets/latex_tables/source_holdout_stress.tex",
        "action": "Use qualitative grids and source-slice diagnostics for the workshop failure-analysis section.",
    },
    {
        "venue_key": "DFF",
        "venue": "DFF-2026 ACM Multimedia workshop",
        "deadline": "2026-07-16",
        "item": "robustness and real-world processing table",
        "status": "ready_asset",
        "paths": "reports/assets/submission_table_robustness_stress.csv; reports/assets/latex_tables/robustness_stress.tex; reports/assets/publication_reverse_transform_robustness.png",
        "action": "Use the transform stress table as the DFF robustness anchor.",
    },
    {
        "venue_key": "DFF",
        "venue": "DFF-2026 ACM Multimedia workshop",
        "deadline": "2026-07-16",
        "item": "bibliography and reproducibility",
        "status": "ready_asset",
        "paths": "references.bib; reports/references_bib_2026_06_14.md; reports/reproducibility_checklist_2026_06_12.md",
        "action": "Verify bibliography metadata and include reproducibility details or appendix links.",
    },
    {
        "venue_key": "DFF",
        "venue": "DFF-2026 ACM Multimedia workshop",
        "deadline": "2026-07-16",
        "item": "final OpenReview upload package",
        "status": "final_export_needed",
        "paths": "reports/assets/paper_skeletons/dff_2026_workshop_skeleton.tex",
        "action": "Compile the DFF PDF, verify anonymous/non-anonymous requirements, and upload through OpenReview.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a final upload checklist for the DFRWS, WIFS, and DFF targets."
    )
    parser.add_argument("--repo-root", default=".", help="Repository root for validating referenced files.")
    parser.add_argument(
        "--out-path",
        default="reports/submission_upload_checklist_2026_06_14.md",
        help="Markdown checklist to write.",
    )
    parser.add_argument(
        "--csv-out",
        default="reports/assets/submission_upload_checklist.csv",
        help="Machine-readable checklist CSV to write.",
    )
    return parser.parse_args()


def _path_list(paths: str) -> list[str]:
    return [part.strip() for part in paths.split(";") if part.strip()]


def _missing_paths(repo_root: Path, paths: str) -> list[str]:
    return [path for path in _path_list(paths) if not (repo_root / path).exists()]


def build_upload_checklist(repo_root: Path) -> tuple[str, pd.DataFrame]:
    repo_root = repo_root.resolve()
    rows = []
    for item in UPLOAD_ITEMS:
        missing = _missing_paths(repo_root, item["paths"])
        row = {
            **item,
            "paths_present": not missing,
            "missing_paths": "; ".join(missing),
        }
        if missing:
            row["status"] = "missing_asset"
        rows.append(row)
    checklist = pd.DataFrame(rows)
    return _write_markdown(checklist), checklist


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


def _venue_summary(checklist: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for venue, group in checklist.groupby("venue", sort=False):
        counts = group["status"].value_counts().to_dict()
        rows.append(
            {
                "venue": venue,
                "deadline": group.iloc[0]["deadline"],
                "ready_assets": counts.get("ready_asset", 0),
                "decision_needed": counts.get("decision_needed", 0),
                "writing_needed": counts.get("writing_needed", 0),
                "final_export_needed": counts.get("final_export_needed", 0),
                "missing_assets": counts.get("missing_asset", 0),
            }
        )
    return pd.DataFrame(rows)


def _write_markdown(checklist: pd.DataFrame) -> str:
    summary = _venue_summary(checklist)
    columns = ["item", "status", "paths", "action", "missing_paths"]
    lines = [
        "# Submission Upload Checklist",
        "",
        "Run date: 2026-06-14",
        "",
        "Generated by `scripts/build_submission_upload_checklist.py` from a venue-specific upload task list and checked-in artifacts.",
        "",
        "This is the practical handoff checklist after the scorecard: it separates checked-in assets from final export, writing, and decision tasks.",
        "",
        "## Venue Summary",
        "",
        _markdown_table(summary, list(summary.columns)),
    ]
    for venue, group in checklist.groupby("venue", sort=False):
        lines.extend(
            [
                "",
                f"## {venue}",
                "",
                f"Deadline: {group.iloc[0]['deadline']}",
                "",
                _markdown_table(group, columns),
            ]
        )
    lines.extend(
        [
            "",
            "## Status Legend",
            "",
            "- `ready_asset`: the needed source artifact is checked in and available.",
            "- `decision_needed`: a human choice remains, usually about which evidence to feature.",
            "- `writing_needed`: generated material exists, but a venue-specific prose draft still needs author editing.",
            "- `final_export_needed`: the source artifact exists, but the venue upload file still needs final formatting/export.",
            "- `missing_asset`: at least one referenced file is missing and should be fixed before upload.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    text, checklist = build_upload_checklist(Path(args.repo_root))
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    csv_path = Path(args.csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    checklist.to_csv(csv_path, index=False)
    print(out_path.resolve())
    print(csv_path.resolve())


if __name__ == "__main__":
    main()
