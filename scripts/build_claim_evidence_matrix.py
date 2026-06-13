from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


COLUMNS = [
    "claim_id",
    "claim",
    "submission_use",
    "status",
    "evidence_finding_ids",
    "evidence_summary",
    "primary_artifact",
    "risk_or_caveat",
    "next_action",
]

CLAIMS = [
    {
        "claim_id": "source_shift_splits_metrics",
        "claim": (
            "Same-domain detector performance is not enough for AI-image forensics; "
            "ranking, calibration, and default decisions diverge under source shift."
        ),
        "submission_use": "DFRWS poster lead; WIFS/DFF introduction and first results table.",
        "status": "ready",
        "evidence_finding_ids": [
            "ishu_same_combined_v3",
            "ishu_same_resnet18",
            "ishu_to_ms_resnet18",
            "ishu_to_ms_scp_fusion_v0",
            "ms_to_ishu_branch_dropout_auc",
            "ms_to_ishu_source_cap_accuracy",
            "ms_to_ishu_source_holdout_tuned_fusion",
            "ms_to_ishu_tuned_fusion_constraint_sweep_best",
            "ms_to_ishu_tuned_fusion_jpeg70",
            "ms_to_ishu_tuned_fusion_blur1",
            "ms_to_ishu_tuned_fusion_resize_half",
            "ms_to_ishu_tuned_fusion_crop85",
            "ms_to_ishu_tuned_fusion_jpeg50",
            "ms_to_ishu_tuned_fusion_noise3",
            "ms_to_ishu_tuned_fusion_screenshot",
        ],
        "primary_artifact": "reports/assets/publication_core_results.md",
        "risk_or_caveat": "Do not frame this as state of the art; frame it as a source-heldout diagnostic benchmark.",
        "next_action": "Keep source-heldout tables in the main paper and use same-domain results only as context.",
    },
    {
        "claim_id": "physics_guided_branch_helps",
        "claim": (
            "Physics-guided ResNet fusion improves over vanilla ResNet and combined_v3 on Ishu same-domain "
            "and gives a useful reverse-transfer calibration anchor."
        ),
        "submission_use": "DFRWS methods/results; WIFS/DFF ablation and interpretability section.",
        "status": "ready_with_caveat",
        "evidence_finding_ids": [
            "ishu_same_combined_v3",
            "ishu_same_resnet18",
            "ishu_same_physics_guided",
            "ishu_to_ms_resnet18",
            "ishu_to_ms_physics_guided",
            "ms_to_ishu_physics_guided",
        ],
        "primary_artifact": "reports/physics_guided_vs_resnet_2026_06_12.md",
        "risk_or_caveat": (
            "This is a single-image physical/signal proxy, not true multi-light photometric stereo; "
            "vanilla ResNet still wins some in-domain MS COCOAI checks."
        ),
        "next_action": "Keep the claim comparative and avoid implying a universal neural upgrade.",
    },
    {
        "claim_id": "clip_transfer_frontier",
        "claim": (
            "Frozen CLIP ViT-B/32 is the strongest standalone transfer-ranking and high-confidence triage branch "
            "in the current benchmark."
        ),
        "submission_use": "DFRWS headline result; DFF foundation-baseline result; WIFS compact comparison.",
        "status": "ready",
        "evidence_finding_ids": [
            "ishu_to_ms_clip_standalone",
            "ishu_to_ms_triage5_clip_standalone",
            "ms_to_ishu_clip_vit_b_32",
        ],
        "primary_artifact": "reports/assets/publication_score_fusion_clip_frontier.png",
        "risk_or_caveat": "CLIP has strong ranking but reverse-transfer default thresholds still over-call generated images.",
        "next_action": "Use CLIP as the foundation baseline to beat; do not bury it inside fusion-only comparisons.",
    },
    {
        "claim_id": "scp_fusion_is_diagnostic",
        "claim": (
            "SCP-Fusion is currently most defensible as a diagnostic fusion protocol: it combines physical, neural, "
            "and foundation signals, but score-level fusion does not always beat the best standalone branch."
        ),
        "submission_use": "DFF method framing; WIFS cautionary fusion result; DFRWS reproducibility panel.",
        "status": "ready_with_caveat",
        "evidence_finding_ids": [
            "ishu_to_ms_scp_fusion_v0",
            "ishu_to_ms_scp_fusion_all_foundation",
            "ishu_to_ms_clip_standalone",
            "ms_to_ishu_score_fusion_all6_temp_balanced",
            "ms_to_ishu_branch_dropout_auc",
            "ms_to_ishu_source_utility_unconstrained",
            "ms_to_ishu_source_holdout_mean_utility_unconstrained",
            "ms_to_ishu_source_holdout_tuned_fusion",
            "ms_to_ishu_tuned_fusion_constraint_sweep_best",
            "ms_to_ishu_tuned_fusion_jpeg70",
            "ms_to_ishu_tuned_fusion_blur1",
            "ms_to_ishu_tuned_fusion_resize_half",
            "ms_to_ishu_tuned_fusion_crop85",
            "ms_to_ishu_tuned_fusion_jpeg50",
            "ms_to_ishu_tuned_fusion_noise3",
            "ms_to_ishu_tuned_fusion_screenshot",
        ],
        "primary_artifact": "reports/assets/publication_reverse_fusion_tradeoff.png",
        "risk_or_caveat": "Current score fusion can suppress the best branch and can inherit source-threshold bias.",
        "next_action": (
            "The source fake-rate constraint sweep improves target accuracy and fake-call bias; "
            "JPEG70, JPEG50, noise, and crop are relatively stable, while blur, resize, and screenshot-style "
            "roundtrips expose the next robustness gap."
        ),
    },
    {
        "claim_id": "source_thresholding_improves_decisions",
        "claim": (
            "Source-aware operating-point selection can turn strong but biased reverse-transfer ranking into better "
            "binary decisions."
        ),
        "submission_use": "DFRWS operational triage panel; WIFS/DFF calibration and utility section.",
        "status": "ready_with_caveat",
        "evidence_finding_ids": [
            "ms_to_ishu_branch_dropout_auc",
            "ms_to_ishu_score_fusion_all6_temp_balanced",
            "ms_to_ishu_source_cap_accuracy",
            "ms_to_ishu_source_utility_cap_0p48",
            "ms_to_ishu_source_holdout_mean_utility_cap_0p48",
            "ms_to_ishu_source_holdout_tuned_fusion",
            "ms_to_ishu_tuned_fusion_constraint_sweep_best",
            "ms_to_ishu_tuned_fusion_jpeg70",
            "ms_to_ishu_tuned_fusion_blur1",
            "ms_to_ishu_tuned_fusion_resize_half",
            "ms_to_ishu_tuned_fusion_crop85",
            "ms_to_ishu_tuned_fusion_jpeg50",
            "ms_to_ishu_tuned_fusion_noise3",
            "ms_to_ishu_tuned_fusion_screenshot",
        ],
        "primary_artifact": "reports/assets/publication_reverse_operating_points.png",
        "risk_or_caveat": "The capped source-threshold result is an operating point, not a learned general solution.",
        "next_action": (
            "Add native-resolution tiling, stronger social-media crops, and larger source splits before calling "
            "the cap frontier contest-ready."
        ),
    },
    {
        "claim_id": "high_confidence_triage_is_viable",
        "claim": (
            "Strict source-heldout two-threshold triage can provide high-accuracy decisions on a subset of target images "
            "instead of forcing unreliable binary calls everywhere."
        ),
        "submission_use": "DFRWS poster workflow; DFF real-world processing discussion.",
        "status": "ready",
        "evidence_finding_ids": [
            "ishu_to_ms_triage5_scp_fusion_all_foundation",
            "ishu_to_ms_triage5_clip_standalone",
        ],
        "primary_artifact": "reports/assets/publication_triage_operating_points.png",
        "risk_or_caveat": "Coverage is intentionally partial; this is an investigative triage mode, not full automation.",
        "next_action": "Add a second qualitative grid for triage failures or reverse-transfer misses.",
    },
    {
        "claim_id": "combined_v4_is_ablation_candidate",
        "claim": (
            "combined_v4 is promising as a richer physical/signal ablation, but it should not be promoted to the main "
            "claimed method until the larger repeated-seed and transfer checks are complete."
        ),
        "submission_use": "WIFS/DFF ablation roadmap; optional appendix if repeated-seed run lands.",
        "status": "needs_more_evidence",
        "evidence_finding_ids": [],
        "primary_artifact": "reports/combined_v4_medium_selectk_probe_2026_06_12.md",
        "risk_or_caveat": "Current v4 evidence is bounded and interval-overlapping; it is not yet a paper headline.",
        "next_action": "Run full repeated-seed raw-v4/select-k60 Ishu and transfer ablations before submission.",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a venue-facing claim/evidence matrix from the publication core table."
    )
    parser.add_argument(
        "--core-results",
        default="reports/assets/publication_core_results.csv",
        help="Generated core result table with finding_id rows.",
    )
    parser.add_argument("--out-dir", default="reports/assets")
    return parser.parse_args()


def _format_metric(row: pd.Series) -> str:
    parts = []
    for source, label in [
        ("accuracy", "acc"),
        ("auc", "AUC"),
        ("brier", "Brier"),
        ("ece", "ECE"),
        ("predicted_fake_rate", "fake-rate"),
        ("coverage", "coverage"),
        ("decided_accuracy", "decided-acc"),
    ]:
        value = row.get(source)
        if pd.notna(value):
            parts.append(f"{label}={float(value):.4f}")
    metrics = ", ".join(parts) if parts else "qualitative/supporting row"
    return f"{row['finding_id']} ({row['method']}: {metrics})"


def _evidence_summary(core_results: pd.DataFrame, finding_ids: list[str]) -> str:
    if not finding_ids:
        return "Tracked outside the generated core result table."
    rows = []
    for finding_id in finding_ids:
        match = core_results[core_results["finding_id"] == finding_id]
        if match.empty:
            raise ValueError(f"Claim references missing finding_id={finding_id!r}")
        rows.append(_format_metric(match.iloc[0]))
    return "; ".join(rows)


def build_claim_matrix(core_results: Path) -> pd.DataFrame:
    core = pd.read_csv(core_results)
    rows = []
    for claim in CLAIMS:
        finding_ids = claim["evidence_finding_ids"]
        rows.append(
            {
                **claim,
                "evidence_finding_ids": ",".join(finding_ids),
                "evidence_summary": _evidence_summary(core, finding_ids),
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def _markdown_escape(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).replace("\n", " ").replace("|", "\\|")


def write_markdown(frame: pd.DataFrame, path: Path) -> None:
    headers = list(frame.columns)
    lines = [
        "# Claim Evidence Matrix",
        "",
        "Generated by `scripts/build_claim_evidence_matrix.py` from `reports/assets/publication_core_results.csv`.",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(_markdown_escape(value) for value in row) + " |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = build_claim_matrix(Path(args.core_results))
    csv_path = out_dir / "claim_evidence_matrix.csv"
    markdown_path = out_dir / "claim_evidence_matrix.md"
    frame.to_csv(csv_path, index=False)
    write_markdown(frame, markdown_path)
    print(csv_path.resolve())
    print(markdown_path.resolve())


if __name__ == "__main__":
    main()
