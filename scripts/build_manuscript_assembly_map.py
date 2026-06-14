from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RUN_DATE = "2026-06-14"

ASSEMBLY_ROWS = [
    {
        "venue": "WIFS",
        "order": 1,
        "section": "Abstract and contribution box",
        "target_pages": 0.25,
        "source_sections": "WIFS compact abstract",
        "primary_assets": "reports/submission_text_drafts_2026_06_14.md; reports/wifs_breadth_decision_2026_06_14.md",
        "writing_action": "Use the compact abstract; keep the contribution wording as source-heldout benchmark evidence, not SOTA.",
    },
    {
        "venue": "WIFS",
        "order": 2,
        "section": "Introduction",
        "target_pages": 0.75,
        "source_sections": "WIFS Introduction Draft",
        "primary_assets": "reports/paper_section_drafts_2026_06_14.md; reports/assets/latex_tables/same_domain_anchor.tex; reports/assets/latex_tables/transfer_frontier.tex",
        "writing_action": "Open with ranking/calibration/threshold shift; cite physics-guided same-domain gain and CLIP transfer frontier.",
    },
    {
        "venue": "WIFS",
        "order": 3,
        "section": "Related work and problem framing",
        "target_pages": 0.75,
        "source_sections": "WIFS Related Work Draft",
        "primary_assets": "references.bib; reports/literature_map_2026_06_14.md",
        "writing_action": "Compress to forensic benchmark families; preserve the single-image physical-proxy caveat.",
    },
    {
        "venue": "WIFS",
        "order": 4,
        "section": "Data, audits, and metrics",
        "target_pages": 0.75,
        "source_sections": "WIFS Data And Audit Draft",
        "primary_assets": "reports/assets/latex_tables/source_holdout_stress.tex; reports/source_holdout_generator_stress_2026_06_14.md",
        "writing_action": "Explain Ishu, source-balanced MS COCOAI, source labels, calibration, fake-call rate, and triage coverage.",
    },
    {
        "venue": "WIFS",
        "order": 5,
        "section": "Methods",
        "target_pages": 1.00,
        "source_sections": "WIFS Methods Draft",
        "primary_assets": "reports/method_family_comparison_2026_06_14.md; reports/assets/latex_tables/method_family_comparison.tex",
        "writing_action": "Describe method families rather than every script; keep SCP-Fusion as a diagnostic protocol.",
    },
    {
        "venue": "WIFS",
        "order": 6,
        "section": "Results",
        "target_pages": 1.75,
        "source_sections": "WIFS Results Draft",
        "primary_assets": "reports/assets/latex_tables/transfer_frontier.tex; reports/assets/latex_tables/reverse_operating_points.tex; reports/assets/latex_tables/robustness_stress.tex; reports/assets/latex_tables/reconstruction_ablation.tex; reports/assets/publication_score_fusion_clip_frontier.png; reports/assets/publication_reverse_operating_points.png",
        "writing_action": "Use compact tables first; include only two or three figures if the official page limit gets tight.",
    },
    {
        "venue": "WIFS",
        "order": 7,
        "section": "Limitations and reproducibility",
        "target_pages": 0.50,
        "source_sections": "Limitations And Reproducibility Draft",
        "primary_assets": "reports/reproducibility_checklist_2026_06_12.md; reports/submission_package_lint_2026_06_14.md",
        "writing_action": "State that raw datasets/models are external while reports, commands, tables, figures, and lints are checked in.",
    },
    {
        "venue": "DFF",
        "order": 1,
        "section": "Abstract and diagnostic claim",
        "target_pages": 0.35,
        "source_sections": "DFF workshop abstract",
        "primary_assets": "reports/submission_text_drafts_2026_06_14.md; reports/method_family_comparison_2026_06_14.md",
        "writing_action": "Frame SCP-Fusion as a diagnostic protocol for source shift, not a universal detector.",
    },
    {
        "venue": "DFF",
        "order": 2,
        "section": "Motivation and related work",
        "target_pages": 1.00,
        "source_sections": "WIFS Introduction Draft; WIFS Related Work Draft",
        "primary_assets": "references.bib; reports/research_deep_dive_2026_06_12.md",
        "writing_action": "Expand the workshop motivation around deepfake forensics, generator shift, and processing robustness.",
    },
    {
        "venue": "DFF",
        "order": 3,
        "section": "Protocol and branches",
        "target_pages": 1.25,
        "source_sections": "WIFS Data And Audit Draft; WIFS Methods Draft",
        "primary_assets": "reports/assets/latex_tables/method_family_comparison.tex; reports/assets/claim_evidence_matrix.md",
        "writing_action": "Keep branch descriptions inspectable: physical/signal, neural, foundation, reconstruction, and score/source fusion.",
    },
    {
        "venue": "DFF",
        "order": 4,
        "section": "Main results and operating points",
        "target_pages": 1.75,
        "source_sections": "WIFS Results Draft",
        "primary_assets": "reports/assets/latex_tables/transfer_frontier.tex; reports/assets/latex_tables/reverse_operating_points.tex; reports/assets/publication_score_fusion_clip_frontier.png; reports/assets/publication_triage_operating_points.png",
        "writing_action": "Carry the multi-objective story: CLIP ranks best, physics helps same-domain, source-aware fusion helps reverse operating points.",
    },
    {
        "venue": "DFF",
        "order": 5,
        "section": "Failure analysis and ablations",
        "target_pages": 1.50,
        "source_sections": "DFF Expansion Draft",
        "primary_assets": "reports/assets/qualitative_seed29_scp_fusion_false_negatives.png; reports/combined_v4_source_slice_diagnostics_2026_06_13.md; reports/assets/latex_tables/source_holdout_stress.tex; reports/assets/latex_tables/reconstruction_ablation.tex",
        "writing_action": "Use the selected qualitative grid and source-slice diagnostics to explain what the fused detector still misses.",
    },
    {
        "venue": "DFF",
        "order": 6,
        "section": "Limitations, ethics, and reproducibility",
        "target_pages": 0.75,
        "source_sections": "Limitations And Reproducibility Draft",
        "primary_assets": "reports/reproducibility_checklist_2026_06_12.md; reports/submission_package_lint_2026_06_14.md",
        "writing_action": "Keep the same overclaim guardrails and call out external datasets/model checkpoints.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a WIFS/DFF manuscript assembly map from checked-in paper artifacts."
    )
    parser.add_argument("--repo-root", default=".", help="Repository root for resolving artifact paths.")
    parser.add_argument(
        "--section-manifest",
        default="reports/assets/paper_section_draft_manifest.csv",
        help="Generated section manifest from build_paper_section_drafts.py.",
    )
    parser.add_argument(
        "--abstract-word-counts",
        default="reports/assets/submission_text_drafts_word_counts.csv",
        help="Generated abstract word-count CSV from build_submission_text_drafts.py.",
    )
    parser.add_argument(
        "--out-path",
        default="reports/manuscript_assembly_map_2026_06_14.md",
        help="Markdown assembly map to write.",
    )
    parser.add_argument(
        "--csv-out",
        default="reports/assets/manuscript_assembly_map.csv",
        help="Machine-readable assembly map to write.",
    )
    return parser.parse_args()


def _path_list(paths: str) -> list[str]:
    return [path.strip() for path in paths.split(";") if path.strip()]


def _paths_exist(repo_root: Path, paths: str) -> bool:
    return all((repo_root / path).exists() for path in _path_list(paths))


def _missing_paths(repo_root: Path, paths: str) -> str:
    missing = [path for path in _path_list(paths) if not (repo_root / path).exists()]
    return "; ".join(missing)


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


def _section_word_counts(section_manifest: Path, abstract_word_counts: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if section_manifest.exists():
        frame = pd.read_csv(section_manifest)
        counts.update(dict(zip(frame["section"], frame["word_count"], strict=True)))
    if abstract_word_counts.exists():
        frame = pd.read_csv(abstract_word_counts)
        counts.update(dict(zip(frame["draft"], frame["word_count"], strict=True)))
    return counts


def _source_word_count(source_sections: str, word_counts: dict[str, int]) -> int:
    total = 0
    for section in [part.strip() for part in source_sections.split(";") if part.strip()]:
        total += int(word_counts.get(section, 0))
    return total


def build_assembly_map(
    repo_root: Path,
    section_manifest: Path,
    abstract_word_counts: Path,
) -> tuple[str, pd.DataFrame]:
    word_counts = _section_word_counts(section_manifest, abstract_word_counts)
    frame = pd.DataFrame(ASSEMBLY_ROWS)
    frame["source_word_count"] = frame["source_sections"].map(
        lambda source_sections: _source_word_count(source_sections, word_counts)
    )
    frame["assets_exist"] = frame["primary_assets"].map(lambda paths: _paths_exist(repo_root, paths))
    frame["missing_assets"] = frame["primary_assets"].map(lambda paths: _missing_paths(repo_root, paths))
    frame = frame.sort_values(["venue", "order"])

    venue_summary = (
        frame.groupby("venue", as_index=False)
        .agg(
            sections=("section", "count"),
            target_pages=("target_pages", "sum"),
            source_word_count=("source_word_count", "sum"),
            assets_present=("assets_exist", "sum"),
        )
        .sort_values("venue")
    )
    venue_summary["assets_total"] = frame.groupby("venue")["assets_exist"].count().values

    lines = [
        "# Manuscript Assembly Map",
        "",
        f"Run date: {RUN_DATE}",
        "",
        "Generated bridge from the checked-in section drafts, paper skeletons, figures, and LaTeX tables to concrete WIFS/DFF writing tasks.",
        "",
        "## Venue Summary",
        "",
        _markdown_table(
            venue_summary,
            ["venue", "sections", "target_pages", "source_word_count", "assets_present", "assets_total"],
        ),
        "",
        "## WIFS Assembly",
        "",
        _markdown_table(
            frame[frame["venue"] == "WIFS"],
            ["order", "section", "target_pages", "source_sections", "source_word_count", "primary_assets", "writing_action"],
        ),
        "",
        "## DFF Assembly",
        "",
        _markdown_table(
            frame[frame["venue"] == "DFF"],
            ["order", "section", "target_pages", "source_sections", "source_word_count", "primary_assets", "writing_action"],
        ),
        "",
        "## Guardrails",
        "",
        "- WIFS should stay compact and multi-objective: ranking, calibration, fake-call rate, source-aware decisions, robustness, and reproducibility.",
        "- DFF can spend more room on failure analysis, source slices, and SCP-Fusion as a diagnostic protocol.",
        "- Do not add new model claims unless they already appear in the checked-in claim-evidence matrix and lints.",
        "- Keep combined_v4 and reconstruction_v2 as ablation evidence unless a later report explicitly promotes them.",
        "",
    ]
    return "\n".join(lines), frame


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    text, frame = build_assembly_map(
        repo_root,
        repo_root / args.section_manifest,
        repo_root / args.abstract_word_counts,
    )

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")

    csv_out = Path(args.csv_out)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_out, index=False)

    print(out_path.resolve())
    print(csv_out.resolve())


if __name__ == "__main__":
    main()
