from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


PAPER_SPECS = [
    {
        "paper_id": "wifs_2026",
        "venue": "IEEE WIFS 2026",
        "title": "Source-Heldout Evaluation of Physical, Neural, and Frozen-Encoder Signals for AI-Generated Image Detection",
        "abstract_header": "WIFS Compact Abstract",
        "filename": "wifs_2026_paper_skeleton.tex",
        "template_hint": "IEEEtran conference skeleton; replace with official WIFS template before submission.",
        "documentclass": r"\documentclass[conference]{IEEEtran}",
        "bibliography_style": "IEEEtran",
        "claim_filter": "WIFS",
    },
    {
        "paper_id": "dff_2026",
        "venue": "DFF-2026 ACM Multimedia Workshop",
        "title": "SCP-Fusion: Source-Calibrated Physical and Foundation Features for Robust AI-Generated Image Forensics",
        "abstract_header": "DFF Workshop Abstract",
        "filename": "dff_2026_workshop_skeleton.tex",
        "template_hint": "ACM-style workshop skeleton; replace with official DFF/ACM template before submission.",
        "documentclass": r"\documentclass[sigconf,review,anonymous]{acmart}",
        "bibliography_style": "ACM-Reference-Format",
        "claim_filter": "DFF",
    },
]

TABLE_INPUTS = [
    ("reports/assets/latex_tables/same_domain_anchor.tex", "same-domain anchor table"),
    ("reports/assets/latex_tables/transfer_frontier.tex", "transfer frontier table"),
    ("reports/assets/latex_tables/reverse_operating_points.tex", "reverse operating point table"),
    ("reports/assets/latex_tables/robustness_stress.tex", "robustness stress table"),
]

FIGURE_INPUTS = [
    (
        "reports/assets/publication_score_fusion_clip_frontier.png",
        "CLIP transfer frontier and all-foundation SCP-Fusion comparison.",
        "clip-frontier",
    ),
    (
        "reports/assets/source_holdout_generator_stress.png",
        "Held-out-generator stress for the source-capped policy; SD3 is currently the weakest source family.",
        "source-stress",
    ),
    (
        "reports/assets/publication_triage_operating_points.png",
        "High-confidence source-heldout triage operating points.",
        "triage-operating-points",
    ),
    (
        "reports/assets/publication_reverse_operating_points.png",
        "Reverse-direction source-aware operating points.",
        "reverse-operating-points",
    ),
    (
        "reports/assets/qualitative_seed17_scp_fusion_false_negatives.png",
        "Representative generated-image misses for failure analysis.",
        "false-negative-grid",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build WIFS/DFF LaTeX paper skeletons from generated submission text and tables."
    )
    parser.add_argument(
        "--text-drafts",
        default="reports/submission_text_drafts_2026_06_14.md",
        help="Generated submission text draft report.",
    )
    parser.add_argument("--out-dir", default="reports/assets/paper_skeletons")
    parser.add_argument(
        "--claim-matrix",
        default="reports/assets/claim_evidence_matrix.csv",
        help="Generated claim/evidence matrix used to add paper-facing claim guardrails.",
    )
    parser.add_argument(
        "--literature-map",
        default="reports/assets/literature_map.csv",
        help="Generated literature map used to seed related-work citations.",
    )
    parser.add_argument(
        "--section-drafts",
        default="reports/paper_section_drafts_2026_06_14.md",
        help="Generated paper section draft report used to populate manuscript prose.",
    )
    parser.add_argument(
        "--report-out",
        default="reports/submission_paper_skeletons_2026_06_14.md",
        help="Markdown report listing generated paper skeletons.",
    )
    return parser.parse_args()


def _extract_section(text: str, header: str) -> str:
    pattern = rf"^## {re.escape(header)}\s*$"
    match = re.search(pattern, text, flags=re.MULTILINE)
    if not match:
        raise ValueError(f"Missing section header: {header}")
    start = match.end()
    next_match = re.search(r"^## ", text[start:], flags=re.MULTILINE)
    end = start + next_match.start() if next_match else len(text)
    return text[start:end].strip()


def _latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "`": "'",
    }
    return "".join(replacements.get(char, char) for char in text)


def _figure_block(path: str, caption: str, label_suffix: str) -> list[str]:
    return [
        r"\begin{figure}[t]",
        r"\centering",
        rf"\includegraphics[width=\linewidth]{{{path}}}",
        rf"\caption{{{_latex_escape(caption)}}}",
        rf"\label{{fig:{label_suffix}}}",
        r"\end{figure}",
        "",
    ]


def _paper_claims(claim_matrix: pd.DataFrame, venue_key: str) -> pd.DataFrame:
    status_mask = claim_matrix["status"].isin(["ready", "ready_with_caveat"])
    venue_mask = claim_matrix["submission_use"].str.contains(venue_key, case=False, na=False)
    return claim_matrix[status_mask & venue_mask].copy()


def _claim_checklist_block(claims: pd.DataFrame) -> list[str]:
    lines = [
        r"\section{Claim-Evidence Checklist}",
        _latex_escape(
            "This generated checklist mirrors the repository claim-evidence matrix so paper claims stay tied "
            "to concrete artifacts and caveats during editing."
        ),
        "",
        r"\begin{itemize}",
    ]
    for claim in claims.itertuples(index=False):
        lines.extend(
            [
                r"\item "
                + rf"\textbf{{{_latex_escape(claim.claim_id)}}}: "
                + _latex_escape(str(claim.claim))
                + " "
                + rf"\textit{{Evidence artifact:}} {_latex_escape(str(claim.primary_artifact))}. "
                + rf"\textit{{Caveat:}} {_latex_escape(str(claim.risk_or_caveat))}",
            ]
        )
    lines.extend([r"\end{itemize}", ""])
    return lines


def _require_literature_keys(literature: pd.DataFrame, keys: list[str]) -> None:
    available = set(literature["key"].astype(str))
    missing = [key for key in keys if key not in available]
    if missing:
        raise ValueError(f"Literature map is missing citation keys: {missing}")


def _cite(keys: list[str]) -> str:
    return r"\cite{" + ",".join(keys) + "}"


def _related_work_block(literature: pd.DataFrame) -> list[str]:
    foundation_keys = [
        "universal_fake_detectors_2023",
        "genimage_2023",
        "aide_chameleon_2025",
        "realhd_2026",
        "bias_free_training_2025",
    ]
    reconstruction_keys = [
        "dire_2023",
        "aeroblade_2024",
        "fire_2025",
        "spectral_any_resolution_2025",
        "no_pixel_left_behind_2025",
        "fake_or_jpeg_2024",
    ]
    physics_keys = ["photometric_faces_2023", "light2lie_2026"]
    _require_literature_keys(literature, foundation_keys + reconstruction_keys + physics_keys)
    return [
        r"\section{Related Work}",
        _latex_escape(
            "Generalization studies and large benchmarks motivate source-heldout evaluation rather than "
            "closed-set detector reporting alone"
        )
        + " "
        + _cite(foundation_keys)
        + ".",
        "",
        _latex_escape(
            "Reconstruction, spectral, high-resolution, and compression-bias studies motivate the "
            "combined_v4/AEROBLADE-lite roadmap, transform stress tests, and bounded native-tiling diagnostics"
        )
        + " "
        + _cite(reconstruction_keys)
        + ".",
        "",
        _latex_escape(
            "Physics-informed generated-image work motivates keeping the single-image physical/signal branch "
            "while avoiding claims of true multi-light photometric stereo or reflectance-law estimation"
        )
        + " "
        + _cite(physics_keys)
        + ".",
        "",
    ]


def _section_draft_map(section_drafts: Path) -> dict[str, str]:
    text = section_drafts.read_text(encoding="utf-8")
    headers = [
        "WIFS Introduction Draft",
        "WIFS Data And Audit Draft",
        "WIFS Methods Draft",
        "WIFS Results Draft",
        "DFF Expansion Draft",
        "Limitations And Reproducibility Draft",
    ]
    return {header: _extract_section(text, header) for header in headers}


def _draft_lines(drafts: dict[str, str], header: str) -> list[str]:
    return [_latex_escape(drafts[header]), ""]


def _paper_tex(
    spec: dict[str, str],
    abstract: str,
    claims: pd.DataFrame,
    literature: pd.DataFrame,
    drafts: dict[str, str],
) -> str:
    lines = [
        "% Auto-generated draft skeleton. Replace the document class with the official venue template before submission.",
        spec["documentclass"],
        r"\usepackage{graphicx}",
        r"\usepackage{booktabs}",
        r"\usepackage{url}",
        "",
        rf"\title{{{_latex_escape(spec['title'])}}}",
        r"\author{Anonymous Author(s)}",
        r"\begin{document}",
        r"\maketitle",
        "",
        r"\begin{abstract}",
        _latex_escape(abstract),
        r"\end{abstract}",
        "",
        r"\section{Introduction}",
    ]
    lines.extend(_draft_lines(drafts, "WIFS Introduction Draft"))
    lines.extend(_related_work_block(literature))
    lines.extend(
        [
            r"\section{Data and Audit}",
        ]
    )
    lines.extend(_draft_lines(drafts, "WIFS Data And Audit Draft"))
    lines.extend(
        [
            r"\section{Methods}",
        ]
    )
    lines.extend(_draft_lines(drafts, "WIFS Methods Draft"))
    lines.extend(
        [
            r"\section{Experiments}",
            _latex_escape(
                "Experiments are organized around repeated seeds, same-domain anchors, cross-domain transfer, "
                "source-heldout generator validation, calibration metrics, high-confidence triage, and robustness "
                "transforms. Tables below are generated from the checked-in result assets."
            ),
            "",
            r"\section{Results}",
        ]
    )
    lines.extend(_draft_lines(drafts, "WIFS Results Draft"))
    lines.extend(
        [
            _latex_escape(
                "The following generated table fragments are checked into the repository and can be edited for space."
            ),
            "",
        ]
    )
    for path, description in TABLE_INPUTS:
        lines.extend([f"% {description}", rf"\input{{{path}}}", ""])
    lines.extend(
        [
            r"\section{Figures and Failure Analysis}",
            _latex_escape(
                "The figure package emphasizes transfer ranking, high-confidence triage, reverse operating "
                "points, held-out generator stress, and generated-image misses. The final venue draft should keep "
                "two or three main figures and move the rest to an appendix if space is tight."
            ),
            "",
            r"Figure~\ref{fig:source-stress} "
            + _latex_escape(
                "turns the source-holdout audit into a manuscript-facing stress view, making the SD3 weak point "
                "visible beside aggregate transfer and robustness numbers."
            ),
            "",
        ]
    )
    for path, caption, label_suffix in FIGURE_INPUTS:
        lines.extend(_figure_block(path, caption, label_suffix))
    if spec["paper_id"] == "dff_2026":
        lines.extend([r"\section{DFF Workshop Expansion}"])
        lines.extend(_draft_lines(drafts, "DFF Expansion Draft"))
    lines.extend(_claim_checklist_block(claims))
    lines.extend(
        [
            r"\section{Limitations}",
        ]
    )
    lines.extend(_draft_lines(drafts, "Limitations And Reproducibility Draft"))
    lines.extend(
        [
            r"\section{Reproducibility}",
            _latex_escape(
                "The public repository includes the submission packet manifest, paper-section draft manifest, "
                "lint reports, dataset/export commands, generated tables, figure builders, and draft bibliography. "
                "Before submission, replace this scaffold with the official venue template and verify all BibTeX "
                "metadata."
            ),
            "",
            r"\bibliographystyle{" + spec["bibliography_style"] + "}",
            r"\bibliography{references}",
            r"\end{document}",
            "",
        ]
    )
    return "\n".join(lines)


def build_paper_skeletons(
    text_drafts: Path,
    claim_matrix_path: Path,
    literature_map_path: Path,
    section_drafts_path: Path,
    out_dir: Path,
) -> pd.DataFrame:
    text = text_drafts.read_text(encoding="utf-8")
    claim_matrix = pd.read_csv(claim_matrix_path)
    literature = pd.read_csv(literature_map_path)
    drafts = _section_draft_map(section_drafts_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in PAPER_SPECS:
        abstract = _extract_section(text, spec["abstract_header"])
        claims = _paper_claims(claim_matrix, spec["claim_filter"])
        tex = _paper_tex(spec, abstract, claims, literature, drafts)
        out_path = out_dir / spec["filename"]
        out_path.write_text(tex, encoding="utf-8")
        draft_section_count = 5 + int(spec["paper_id"] == "dff_2026")
        rows.append(
            {
                "paper_id": spec["paper_id"],
                "venue": spec["venue"],
                "title": spec["title"],
                "path": out_path.as_posix(),
                "template_hint": spec["template_hint"],
                "abstract_header": spec["abstract_header"],
                "claim_count": len(claims),
                "citation_count": len(re.findall(r"\\cite\{", tex)),
                "draft_section_count": draft_section_count,
                "todo_count": len(re.findall(r"\bTODO\b", tex)),
            }
        )
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_dir / "submission_paper_skeleton_manifest.csv", index=False)
    return manifest


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(str(value).replace("\n", " ") for value in row) + " |")
    return "\n".join(lines)


def write_report(manifest: pd.DataFrame, report_out: Path) -> None:
    lines = [
        "# Submission Paper Skeletons",
        "",
        "Run date: 2026-06-14",
        "",
        "Generated by `scripts/build_submission_paper_skeletons.py` from generated submission text, section drafts, LaTeX table fragments, the claim-evidence matrix, and the literature map.",
        "",
        "These files are draft scaffolds, not official venue templates. Replace the document class and formatting with the final WIFS/DFF instructions before submission.",
        "",
        _markdown_table(manifest),
        "",
    ]
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    manifest = build_paper_skeletons(
        Path(args.text_drafts),
        Path(args.claim_matrix),
        Path(args.literature_map),
        Path(args.section_drafts),
        Path(args.out_dir),
    )
    write_report(manifest, Path(args.report_out))
    print(Path(args.report_out).resolve())
    print((Path(args.out_dir) / "submission_paper_skeleton_manifest.csv").resolve())


if __name__ == "__main__":
    main()
