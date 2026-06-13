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
    ),
    (
        "reports/assets/publication_triage_operating_points.png",
        "High-confidence source-heldout triage operating points.",
    ),
    (
        "reports/assets/publication_reverse_operating_points.png",
        "Reverse-direction source-aware operating points.",
    ),
    (
        "reports/assets/qualitative_seed17_scp_fusion_false_negatives.png",
        "Representative generated-image misses for failure analysis.",
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


def _paper_tex(spec: dict[str, str], abstract: str, claims: pd.DataFrame) -> str:
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
        _latex_escape(
            "TODO: Motivate source-heldout AI-generated image forensics and explain why ranking, "
            "calibration, fake-call rate, and triage coverage should be reported separately."
        ),
        "",
        r"\section{Related Work}",
        _latex_escape(
            "TODO: Cover AI-generated image detection, single-image physical/signal forensics, "
            "frozen foundation encoders, score fusion, calibration, and forensic triage."
        ),
        "",
        r"\section{Data and Audit}",
        _latex_escape(
            "TODO: Describe Ishu AI-vs-real, source-balanced MS COCOAI, generator source labels, "
            "dataset export commands, and duplicate/leakage audit assumptions."
        ),
        "",
        r"\section{Methods}",
        _latex_escape(
            "TODO: Define combined_v3/v4 physical-signal features, ResNet-18, physics-guided "
            "ResNet fusion, frozen ConvNeXt/DINOv2/CLIP probes, SCP-Fusion, source-heldout "
            "thresholding, and high-confidence triage."
        ),
        "",
        r"\section{Experiments}",
        _latex_escape(
            "TODO: State repeated seeds, same-domain and transfer splits, source-heldout "
            "generator validation, calibration metrics, and robustness transforms."
        ),
        "",
        r"\section{Results}",
        _latex_escape(
            "The following generated table fragments are checked into the repository and can be edited for space."
        ),
        "",
    ]
    for path, description in TABLE_INPUTS:
        lines.extend([f"% {description}", rf"\input{{{path}}}", ""])
    lines.extend(
        [
            r"\section{Figures and Failure Analysis}",
            _latex_escape("TODO: Select two to three figures for the main paper and move the rest to an appendix."),
            "",
        ]
    )
    for index, (path, caption) in enumerate(FIGURE_INPUTS, start=1):
        lines.extend(_figure_block(path, caption, f"submission-figure-{index}"))
    lines.extend(_claim_checklist_block(claims))
    lines.extend(
        [
            r"\section{Limitations}",
            _latex_escape(
                "TODO: Keep these caveats explicit: the physical branch is a single-image proxy, "
                "SCP-Fusion does not universally beat CLIP, native tiling only changes the conventional "
                "target branch in the current fused diagnostic, and robustness depends on the transform."
            ),
            "",
            r"\section{Reproducibility}",
            _latex_escape(
                "TODO: Cite the public repository, submission packet manifest, lint report, dataset "
                "export commands, and generated table/figure builders."
            ),
            "",
            r"\bibliographystyle{" + spec["bibliography_style"] + "}",
            r"\bibliography{references}",
            r"\end{document}",
            "",
        ]
    )
    return "\n".join(lines)


def build_paper_skeletons(text_drafts: Path, claim_matrix_path: Path, out_dir: Path) -> pd.DataFrame:
    text = text_drafts.read_text(encoding="utf-8")
    claim_matrix = pd.read_csv(claim_matrix_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in PAPER_SPECS:
        abstract = _extract_section(text, spec["abstract_header"])
        claims = _paper_claims(claim_matrix, spec["claim_filter"])
        tex = _paper_tex(spec, abstract, claims)
        out_path = out_dir / spec["filename"]
        out_path.write_text(tex, encoding="utf-8")
        rows.append(
            {
                "paper_id": spec["paper_id"],
                "venue": spec["venue"],
                "title": spec["title"],
                "path": out_path.as_posix(),
                "template_hint": spec["template_hint"],
                "abstract_header": spec["abstract_header"],
                "claim_count": len(claims),
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
        "Generated by `scripts/build_submission_paper_skeletons.py` from generated submission text, LaTeX table fragments, and the claim-evidence matrix.",
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
    manifest = build_paper_skeletons(Path(args.text_drafts), Path(args.claim_matrix), Path(args.out_dir))
    write_report(manifest, Path(args.report_out))
    print(Path(args.report_out).resolve())
    print((Path(args.out_dir) / "submission_paper_skeleton_manifest.csv").resolve())


if __name__ == "__main__":
    main()
