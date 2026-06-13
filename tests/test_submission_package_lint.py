from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_submission_package_lint_passes_clean_generated_packet(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    required_paths = [
        "README.md",
        "reports/submission_text_drafts_2026_06_14.md",
        "reports/assets/submission_text_drafts_word_counts.csv",
    ]
    for relative in required_paths:
        path = repo_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder\n", encoding="utf-8")

    manifest = tmp_path / "submission_packet_manifest.csv"
    text_drafts = tmp_path / "submission_text_drafts.md"
    word_counts = tmp_path / "submission_text_drafts_word_counts.csv"
    out_path = tmp_path / "submission_package_lint.md"
    checks_out = tmp_path / "submission_package_lint.csv"

    pd.DataFrame(
        {
            "path": required_paths,
            "type": ["repo", "writing", "writing"],
            "venues": ["DFRWS,WIFS,DFF", "DFRWS,WIFS,DFF", "DFRWS,WIFS,DFF"],
            "purpose": ["purpose"] * 3,
            "required": [True, True, True],
            "exists": [True, True, True],
            "size_bytes": [12, 12, 12],
        }
    ).to_csv(manifest, index=False)

    dfrws = (
        "single-image physical proxy keeps the claim careful. CLIP as the current ranking frontier is stated. "
        "SCP-Fusion as a diagnostic protocol is stated. native tiling as a bounded conventional-branch result is stated. "
        "ishu_same_physics_guided ishu_to_ms_clip_standalone ishu_to_ms_triage5_clip_standalone "
        "ms_to_ishu_tuned_fusion_constraint_sweep_best ms_to_ishu_tuned_fusion_native_tiling_best "
        "ms_to_ishu_tuned_fusion_jpeg30 ms_to_ishu_tuned_fusion_blur1. "
        "This abstract repeats careful source-heldout forensic wording with ranking calibration triage transforms "
        "dataset bias and failure analysis enough times to satisfy the lower word bound without unsafe claims. "
        "The poster framing remains public reproducibility, evidence traceability, conservative interpretation, "
        "auditable metrics, and reviewer-facing caveats."
    )
    wifs = (
        "This compact abstract discusses source-heldout evaluation, ranking, calibration, operating points, "
        "physical signal features, foundation encoders, triage, robustness transforms, and careful caveats. "
        "It avoids unsafe superiority language while preserving enough methodological detail for a paper abstract. "
        "The wording also mentions repeated seeds, reverse transfer, source thresholds, missing production claims, "
        "and separate reporting of accuracy, AUC, calibration, and fake-call behavior. These details make the "
        "fixture long enough to test the same lower bound used for real drafts while staying narrowly scoped today."
    )
    dff = (
        "This workshop abstract frames SCP-Fusion as a diagnostic benchmark for robustness, dataset bias, "
        "calibration, triage, foundation encoders, physical signal features, and qualitative failure analysis. "
        "It keeps the claims bounded and avoids universal detector language while remaining publication ready. "
        "The text also names real-world processing, source-slice diagnostics, transform stress tests, and public "
        "artifact tracking so the linter can verify a realistic workshop-length abstract without unsafe phrases. "
        "Additional wording covers reproducible tables, calibrated thresholds, partial decisions, failure grids, and caveats for reviewers."
    )
    text_drafts.write_text(
        "\n".join(
            [
                "# Submission Text Drafts",
                "",
                "## DFRWS Poster Abstract",
                "",
                dfrws,
                "",
                "## WIFS Compact Abstract",
                "",
                wifs,
                "",
                "## DFF Workshop Abstract",
                "",
                dff,
                "",
            ]
        ),
        encoding="utf-8",
    )
    pd.DataFrame(
        {
            "draft": ["DFRWS poster abstract", "WIFS compact abstract", "DFF workshop abstract"],
            "word_count": [_word_count(dfrws), _word_count(wifs), _word_count(dff)],
        }
    ).to_csv(word_counts, index=False)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "lint_submission_package.py"),
            "--repo-root",
            str(repo_root),
            "--manifest",
            str(manifest),
            "--text-drafts",
            str(text_drafts),
            "--word-counts",
            str(word_counts),
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
    assert "abstracts avoid overclaim" in checks["check"].str.cat(sep=" ")


def _word_count(text: str) -> int:
    import re

    return len(re.findall(r"\b[\w'-]+\b", text))
