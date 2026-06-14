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
            "ms_to_ishu_tuned_fusion_native_tiling_best",
            "ms_to_ishu_tuned_fusion_jpeg70",
            "ms_to_ishu_tuned_fusion_blur1",
            "ms_to_ishu_tuned_fusion_resize_half",
            "ms_to_ishu_tuned_fusion_crop85",
            "ms_to_ishu_tuned_fusion_jpeg50",
            "ms_to_ishu_tuned_fusion_jpeg30",
            "ms_to_ishu_tuned_fusion_noise3",
            "ms_to_ishu_tuned_fusion_screenshot",
            "ms_to_ishu_tuned_fusion_social_square",
            "ms_to_ishu_tuned_fusion_social_720p",
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
            "ms_to_ishu_tuned_fusion_native_tiling_best",
            "ms_to_ishu_tuned_fusion_jpeg70",
            "ms_to_ishu_tuned_fusion_blur1",
            "ms_to_ishu_tuned_fusion_resize_half",
            "ms_to_ishu_tuned_fusion_crop85",
            "ms_to_ishu_tuned_fusion_jpeg50",
            "ms_to_ishu_tuned_fusion_jpeg30",
            "ms_to_ishu_tuned_fusion_noise3",
            "ms_to_ishu_tuned_fusion_screenshot",
            "ms_to_ishu_tuned_fusion_social_square",
            "ms_to_ishu_tuned_fusion_social_720p",
        ],
        "primary_artifact": "reports/assets/publication_reverse_fusion_tradeoff.png",
        "risk_or_caveat": "Current score fusion can suppress the best branch and can inherit source-threshold bias.",
        "next_action": (
            "The source fake-rate constraint sweep improves target accuracy and fake-call bias; "
            "native tiled combined_v3 gives a small fused-stack gain; JPEG70, JPEG50, noise, social-square, "
            "and social-720p processing are relatively stable, while JPEG30, blur, resize, and "
            "screenshot-style roundtrips expose the next robustness gap."
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
            "ms_to_ishu_tuned_fusion_native_tiling_best",
            "ms_to_ishu_tuned_fusion_jpeg70",
            "ms_to_ishu_tuned_fusion_blur1",
            "ms_to_ishu_tuned_fusion_resize_half",
            "ms_to_ishu_tuned_fusion_crop85",
            "ms_to_ishu_tuned_fusion_jpeg50",
            "ms_to_ishu_tuned_fusion_jpeg30",
            "ms_to_ishu_tuned_fusion_noise3",
            "ms_to_ishu_tuned_fusion_screenshot",
            "ms_to_ishu_tuned_fusion_social_square",
            "ms_to_ishu_tuned_fusion_social_720p",
        ],
        "primary_artifact": "reports/assets/publication_reverse_operating_points.png",
        "risk_or_caveat": "The capped source-threshold result is an operating point, not a learned general solution.",
        "next_action": (
            "Use the fused native-tiling diagnostic as a bounded gain; add larger source splits before calling "
            "the cap frontier contest-ready."
        ),
    },
    {
        "claim_id": "transform_stress_exposes_failure_modes",
        "claim": (
            "Reverse tuned-fusion SCP-Fusion has identifiable transform failure modes: half-resolution resize and "
            "blur hurt ranking most, while harsh JPEG hurts default decisions most."
        ),
        "submission_use": "DFRWS robustness panel; WIFS/DFF robustness and failure-analysis section.",
        "status": "ready_with_caveat",
        "evidence_finding_ids": [
            "ms_to_ishu_tuned_fusion_constraint_sweep_best",
            "ms_to_ishu_tuned_fusion_resize_half",
            "ms_to_ishu_tuned_fusion_blur1",
            "ms_to_ishu_tuned_fusion_screenshot",
            "ms_to_ishu_tuned_fusion_jpeg30",
            "ms_to_ishu_tuned_fusion_noise3",
            "ms_to_ishu_tuned_fusion_social_square",
        ],
        "primary_artifact": "reports/robustness_failure_ranking_2026_06_14.md",
        "risk_or_caveat": (
            "This is source-selected proxy transform stress, not an official NTIRE/ImageCLEF challenge score or "
            "a universal robustness claim."
        ),
        "next_action": (
            "Use resize, blur, screenshot, and JPEG30 as failure examples; write noise/social gains as bounded "
            "proxy observations only."
        ),
    },
    {
        "claim_id": "tiled_dino_mode_tradeoff",
        "claim": (
            "Tiled-DINO branch replacement is mode-sensitive: `tile_max` improves fused decision/ranking under "
            "core transform stress, while `tile_mean` is safer for calibration diagnostics."
        ),
        "submission_use": "DFRWS poster follow-up; WIFS/DFF robustness and calibration section.",
        "status": "ready_with_caveat",
        "evidence_finding_ids": [],
        "evidence_summary_source": "tiled_dino_tradeoff",
        "primary_artifact": "reports/tiled_dinov2_calibration_tradeoff_2026_06_14.md",
        "risk_or_caveat": (
            "This is derived from source-selected proxy transform stress, not an official external benchmark; "
            "do not claim tiled-DINO improves calibration universally."
        ),
        "next_action": (
            "Use `tile_max` for decision/ranking robustness headlines, use `tile_mean` for Brier/ECE discussion, "
            "and keep official benchmark claims separate."
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
        "next_action": (
            "Use the seed-17 and seed-29 qualitative grids to illustrate confident misses and branch "
            "disagreement; add a reverse-transfer grid only if space allows."
        ),
    },
    {
        "claim_id": "combined_v4_is_ablation_candidate",
        "claim": (
            "combined_v4 is useful as a richer physical/signal ablation, especially with feature selection, but the "
            "completed transfer gate does not justify replacing combined_v3 as the main conventional baseline."
        ),
        "submission_use": "WIFS/DFF ablation roadmap; appendix feature-family caveat.",
        "status": "ready_with_caveat",
        "evidence_finding_ids": [
            "ishu_same_combined_v4_raw",
            "ishu_same_combined_v4_selectk60",
            "ishu_to_ms_combined_v4_raw",
            "ishu_to_ms_combined_v4_selectk60",
        ],
        "primary_artifact": "reports/combined_v4_full_transfer_summary_2026_06_13.md",
        "risk_or_caveat": (
            "Raw v4 mainly helps transfer accuracy, while select-k60 helps transfer AUC/calibration but loses "
            "same-domain Ishu accuracy; source-slice diagnostics show the gains and losses are generator/category "
            "uneven, so keep it as an ablation rather than a headline method."
        ),
        "next_action": (
            "Try source-aware feature selection or a stronger regularized classifier before any main-method promotion."
        ),
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
    parser.add_argument(
        "--tiled-dino-tradeoff",
        default="reports/assets/tiled_dinov2_calibration_tradeoff.csv",
        help="Tiled-DINO calibration tradeoff table used for artifact-backed claim evidence.",
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


def _bool_count(series: pd.Series) -> int:
    return int(series.astype(str).str.lower().isin(["true", "1", "yes"]).sum())


def _signed_metric(value: float) -> str:
    return f"{float(value):+.4f}"


def _tiled_dino_evidence_summary(tradeoff_path: Path) -> str:
    tradeoff = pd.read_csv(tradeoff_path)
    required = {
        "variant",
        "score_mode",
        "target_accuracy_mean_delta_vs_global",
        "target_roc_auc_mean_delta_vs_global",
        "target_brier_score_mean_improved_vs_global",
        "target_expected_calibration_error_mean_improved_vs_global",
    }
    missing = required - set(tradeoff.columns)
    if missing:
        raise ValueError(f"Missing tiled-DINO tradeoff columns: {sorted(missing)}")
    n_transforms = int(tradeoff["variant"].nunique())
    mode_means = (
        tradeoff.groupby("score_mode", as_index=True)[
            ["target_accuracy_mean_delta_vs_global", "target_roc_auc_mean_delta_vs_global"]
        ]
        .mean()
        .to_dict("index")
    )
    for mode in ["tile_max", "tile_mean"]:
        if mode not in mode_means:
            raise ValueError(f"Missing tiled-DINO score_mode={mode!r}")
    tile_mean = tradeoff[tradeoff["score_mode"].eq("tile_mean")]
    return (
        "tiled_dinov2_calibration_tradeoff "
        f"(tile_max: mean acc_delta={_signed_metric(mode_means['tile_max']['target_accuracy_mean_delta_vs_global'])}, "
        f"mean AUC_delta={_signed_metric(mode_means['tile_max']['target_roc_auc_mean_delta_vs_global'])} "
        f"across {n_transforms} transform-stress probes; "
        f"tile_mean: Brier improves on {_bool_count(tile_mean['target_brier_score_mean_improved_vs_global'])}/{n_transforms}, "
        f"ECE improves on {_bool_count(tile_mean['target_expected_calibration_error_mean_improved_vs_global'])}/{n_transforms})"
    )


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


def build_claim_matrix(core_results: Path, tiled_dino_tradeoff: Path) -> pd.DataFrame:
    core = pd.read_csv(core_results)
    rows = []
    for claim in CLAIMS:
        finding_ids = claim["evidence_finding_ids"]
        if claim.get("evidence_summary_source") == "tiled_dino_tradeoff":
            evidence_summary = _tiled_dino_evidence_summary(tiled_dino_tradeoff)
        else:
            evidence_summary = _evidence_summary(core, finding_ids)
        rows.append(
            {
                **claim,
                "evidence_finding_ids": ",".join(finding_ids),
                "evidence_summary": evidence_summary,
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
        "Generated by `scripts/build_claim_evidence_matrix.py` from `reports/assets/publication_core_results.csv` and artifact-backed diagnostic tables.",
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
    frame = build_claim_matrix(Path(args.core_results), Path(args.tiled_dino_tradeoff))
    csv_path = out_dir / "claim_evidence_matrix.csv"
    markdown_path = out_dir / "claim_evidence_matrix.md"
    frame.to_csv(csv_path, index=False)
    write_markdown(frame, markdown_path)
    print(csv_path.resolve())
    print(markdown_path.resolve())


if __name__ == "__main__":
    main()
