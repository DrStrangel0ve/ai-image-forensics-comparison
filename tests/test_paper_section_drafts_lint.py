from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_paper_section_drafts_lint_passes_required_sections(tmp_path: Path) -> None:
    section_drafts = tmp_path / "paper_section_drafts.md"
    manifest = tmp_path / "paper_section_draft_manifest.csv"
    out_path = tmp_path / "paper_section_drafts_lint.md"
    checks_out = tmp_path / "paper_section_drafts_lint.csv"

    sections = {
        "WIFS Introduction Draft": (
            "This source-heldout introduction tracks ranking, calibration, and threshold behavior under source "
            "shift. It reports 0.8450 for the physics-guided anchor and 0.8641 for the transfer ranker. The text "
            "adds paired-seed support for forensic decisions, fake-call rates, repeated seeds, generator changes, "
            "and cautious evaluation to satisfy the expected range without making broad claims about deployment."
        ),
        "WIFS Related Work Draft": (
            "The related work cites universal_fake_detectors_2023 for frozen encoders and motivates spectral and "
            "compression-aware checks. It explicitly says the physical branch is a single-image physical/signal proxy "
            "rather than calibrated multi-light photometric stereo, preserving the caveat while still connecting the "
            "method to physics-informed image forensics and source-heldout evaluation."
        ),
        "WIFS Data And Audit Draft": (
            "The data section names Ishu AI-vs-real and MS COCOAI, explains exported generator source labels, and "
            "points readers to artifact manifests. It says raw datasets are external while result reports, commands, "
            "tables, and figure assets are public, which keeps the reproducibility scope honest for reviewers."
        ),
        "WIFS Methods Draft": (
            "The methods section defines combined_v3, ResNet-18, physics-guided fusion, and SCP-Fusion. It explains "
            "that the comparison reports AUC, accuracy, Brier score, ECE, fake-call rate, and triage coverage rather "
            "than relying on one metric. The language stays descriptive and avoids overstating the branch fusion."
        ),
        "WIFS Results Draft": (
            "The results section carries 0.8246, 0.8450, 0.8641, and 0.7749 as required metric anchors. It also "
            "mentions JPEG30 and blur as stressors, explaining that robustness is transform-specific. Extra text "
            "describes CLIP ranking, source-capped reverse decisions, native tiling, social processing, and partial "
            "triage so the section has realistic manuscript density. It includes reconstruction_v2 as an ablation "
            "caveat rather than a headline method, plus paired-seed support for conservative result wording."
        ),
        "DFF Expansion Draft": (
            "The DFF expansion frames SCP-Fusion as a diagnostic protocol. It discusses combined_v4 as an ablation "
            "candidate, keeps AEROBLADE/FIRE-style reconstruction as future work, and explains how failure grids can "
            "separate semantic, spectral, compression, and source-threshold artifacts. The paragraph remains cautious "
            "about what is implemented today and names reconstruction_v2 as a source-sensitive ablation. Paired-seed support "
            "is presented as a robustness qualifier."
        ),
        "Limitations And Reproducibility Draft": (
            "The limitations section says the physical branch is a single-image proxy, SCP-Fusion does not universally "
            "beat frozen CLIP, official venue templates are still required, and verified bibliography metadata remains "
            "a final submission task. It also names public tables, figures, lint reports, and command builders as the "
            "current reproducibility surface."
        ),
    }
    filler = (
        "This draft remains conservative about generator shift, evaluation scope, reviewer interpretation, "
        "public artifacts, and remaining submission work."
    )
    sections = {name: text + " " + " ".join([filler] * 5) for name, text in sections.items()}

    lines = ["# Paper Section Drafts", ""]
    for name, text in sections.items():
        lines.extend([f"## {name}", "", text, ""])
    section_drafts.write_text("\n".join(lines), encoding="utf-8")

    pd.DataFrame(
        [
            {
                "section": name,
                "word_count": _word_count(text),
                "has_metric": bool(re.search(r"\d\.\d{4}", text)),
                "has_caveat": "single-image" in text or "does not universally beat" in text,
            }
            for name, text in sections.items()
        ]
    ).to_csv(manifest, index=False)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "lint_paper_section_drafts.py"),
            "--section-drafts",
            str(section_drafts),
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
    assert "required phrase present: 0.8641" in checks["check"].str.cat(sep=" ")
    assert "avoid overclaim" in checks["check"].str.cat(sep=" ")


def _word_count(text: str) -> int:
    return len(re.findall(r"\b[\w'-]+\b", text))
