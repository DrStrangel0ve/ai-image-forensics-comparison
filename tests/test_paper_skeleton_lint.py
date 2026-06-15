from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_paper_skeleton_lint_validates_paths_and_claim_guardrails(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    skeleton_dir = repo_root / "reports" / "assets" / "paper_skeletons"
    table_dir = repo_root / "reports" / "assets" / "latex_tables"
    asset_dir = repo_root / "reports" / "assets"
    skeleton_dir.mkdir(parents=True)
    table_dir.mkdir(parents=True)
    asset_dir.mkdir(parents=True, exist_ok=True)
    (repo_root / "references.bib").write_text(
        "\n".join(
            [
                "@misc{universal_fake_detectors_2023,",
                "  title = {Universal Fake Detectors},",
                "  year = {2023},",
                "}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    table_paths = [
        "reports/assets/latex_tables/method_family_comparison.tex",
        "reports/assets/latex_tables/same_domain_anchor.tex",
        "reports/assets/latex_tables/transfer_frontier.tex",
        "reports/assets/latex_tables/reverse_operating_points.tex",
        "reports/assets/latex_tables/robustness_stress.tex",
        "reports/assets/latex_tables/source_holdout_stress.tex",
        "reports/assets/latex_tables/reconstruction_ablation.tex",
        "reports/assets/latex_tables/calibration_operating_modes.tex",
    ]
    figure_paths = [
        "reports/assets/publication_score_fusion_clip_frontier.png",
        "reports/assets/source_holdout_generator_stress.png",
        "reports/assets/publication_triage_operating_points.png",
        "reports/assets/publication_reverse_operating_points.png",
        "reports/assets/qualitative_seed17_scp_fusion_false_negatives.png",
    ]
    for relative in table_paths + figure_paths:
        path = repo_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder\n", encoding="utf-8")

    tex_path = skeleton_dir / "wifs_2026_paper_skeleton.tex"
    tex_path.write_text(
        "\n".join(
            [
                r"\documentclass[conference]{IEEEtran}",
                r"\title{Skeleton}",
                r"\begin{document}",
                r"\maketitle",
                r"\begin{abstract}",
                "Abstract.",
                r"\end{abstract}",
                r"Related work seed \cite{universal_fake_detectors_2023}.",
                r"Figure~\ref{fig:source-stress} shows the held-out generator stress view.",
                *[rf"\input{{{path}}}" for path in table_paths],
                *[rf"\includegraphics[width=\linewidth]{{{path}}}" for path in figure_paths],
                r"\label{fig:source-stress}",
                r"\section{Claim-Evidence Checklist}",
                r"\begin{itemize}",
                (
                    r"\item \textbf{claim_one}: Claim text. "
                    r"\textit{Evidence artifact:} reports/example.md. "
                    r"\textit{Caveat:} single-image proxy; does not universally beat CLIP; "
                    "robustness depends on the transform."
                ),
                r"\end{itemize}",
                r"\bibliography{references}",
                r"\end{document}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest = skeleton_dir / "submission_paper_skeleton_manifest.csv"
    pd.DataFrame(
        [
            {
                "paper_id": "wifs_2026",
                "venue": "IEEE WIFS 2026",
                "title": "Skeleton",
                "path": "reports/assets/paper_skeletons/wifs_2026_paper_skeleton.tex",
                "template_hint": "IEEEtran",
                "abstract_header": "WIFS Compact Abstract",
                "claim_count": 1,
                "citation_count": 1,
                "draft_section_count": 5,
                "todo_count": 0,
            }
        ]
    ).to_csv(manifest, index=False)

    out_path = tmp_path / "paper_skeleton_lint.md"
    checks_out = tmp_path / "paper_skeleton_lint.csv"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "lint_paper_skeletons.py"),
            "--repo-root",
            str(repo_root),
            "--manifest",
            str(manifest),
            "--out-path",
            str(out_path),
            "--checks-out",
            str(checks_out),
        ],
        cwd=ROOT,
        check=True,
    )

    report = out_path.read_text(encoding="utf-8")
    checks = pd.read_csv(checks_out)
    assert "Status: **PASS**" in report
    assert checks["passed"].all()
    assert "claim count matches manifest" in checks["check"].str.cat(sep=" ")
    assert "citation keys exist in references.bib" in checks["check"].str.cat(sep=" ")
    assert "no TODO placeholders" in checks["check"].str.cat(sep=" ")
    assert "method family table referenced" in checks["check"].str.cat(sep=" ")
    assert "source stress table referenced" in checks["check"].str.cat(sep=" ")
    assert "reconstruction ablation table referenced" in checks["check"].str.cat(sep=" ")
    assert "operating-mode table referenced" in checks["check"].str.cat(sep=" ")
    assert "source stress figure referenced" in checks["check"].str.cat(sep=" ")
