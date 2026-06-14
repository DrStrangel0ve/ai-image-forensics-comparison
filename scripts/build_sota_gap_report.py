from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RUN_DATE = "2026-06-14"

LOCAL_FRONTIER_IDS = [
    "ishu_same_physics_guided",
    "ishu_to_ms_clip_standalone",
    "ishu_to_ms_triage5_clip_standalone",
    "ms_to_ishu_tuned_fusion_native_tiling_best",
    "ms_to_ishu_tuned_fusion_constraint_sweep_best",
    "ms_to_ishu_tuned_fusion_resize_half",
    "ms_to_ishu_tuned_fusion_blur1",
    "ms_to_ishu_tuned_fusion_noise3",
]

SOTA_ANCHORS = [
    {
        "benchmark_id": "ntire_2026_robust_aigc",
        "benchmark": "NTIRE 2026 Robust AI-Generated Image Detection in the Wild",
        "status": "official_sota_available_not_submitted",
        "official_metric": "average Robust ROC-AUC",
        "official_best_method": "MICV",
        "official_best_score": 0.9723,
        "official_clean_reference": 0.9974,
        "comparison_validity": "not_apples_to_apples",
        "reason": "Official NTIRE uses a hidden/open challenge test set with 42 generators and 36 transforms; this repo has only proxy Ishu/MS COCOAI transfer and transform stress tests.",
        "source_url": "https://openaccess.thecvf.com/content/CVPR2026W/NTIRE/papers/Gushchin_NTIRE_2026_Challenge_on_Robust_AI-Generated_Image_Detection_in_the_CVPRW_2026_paper.pdf",
    },
    {
        "benchmark_id": "imageclef_2026_deepfake",
        "benchmark": "ImageCLEF 2026 Deepfake Detection and Generation",
        "status": "official_task_closed_not_submitted",
        "official_metric": "accuracy/recall/precision/F1",
        "official_best_method": "not_checked_in",
        "official_best_score": pd.NA,
        "official_clean_reference": pd.NA,
        "comparison_validity": "no_official_score",
        "reason": "The repo has no ImageCLEF task data or run submission; source-heldout triage is protocol-inspired only.",
        "source_url": "https://www.imageclef.org/2026/deepfake-detection-and-generation",
    },
    {
        "benchmark_id": "genimage_chameleon_academic",
        "benchmark": "GenImage / Chameleon academic benchmark family",
        "status": "not_run_locally",
        "official_metric": "cross-generator/degraded accuracy or detector-specific metrics",
        "official_best_method": "AIDE-style hybrid feature detectors and later robust/reconstruction detectors",
        "official_best_score": pd.NA,
        "official_clean_reference": pd.NA,
        "comparison_validity": "methodological_only",
        "reason": "GenImage and Chameleon are relevant SOTA references, but this repo has not run their official splits.",
        "source_url": "https://genimage-dataset.github.io/; https://arxiv.org/abs/2406.19435",
    },
]

NTIRE_PROXY_IDS = [
    "ms_to_ishu_tuned_fusion_constraint_sweep_best",
    "ms_to_ishu_tuned_fusion_native_tiling_best",
    "ms_to_ishu_tuned_fusion_resize_half",
    "ms_to_ishu_tuned_fusion_blur1",
    "ms_to_ishu_tuned_fusion_noise3",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a guarded SOTA-gap report from local publication metrics and external benchmark metadata."
    )
    parser.add_argument(
        "--core-results",
        default="reports/assets/publication_core_results.csv",
        help="Canonical local metric table.",
    )
    parser.add_argument(
        "--out-path",
        default="reports/sota_gap_report_2026_06_14.md",
        help="Markdown report to write.",
    )
    parser.add_argument(
        "--csv-out",
        default="reports/assets/sota_gap_report.csv",
        help="Machine-readable SOTA gap rows to write.",
    )
    return parser.parse_args()


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


def _format_metric(row: pd.Series) -> str:
    parts = []
    for column, label in [
        ("accuracy", "acc"),
        ("auc", "AUC"),
        ("brier", "Brier"),
        ("ece", "ECE"),
        ("predicted_fake_rate", "fake-call"),
        ("coverage", "coverage"),
        ("decided_accuracy", "decided-acc"),
    ]:
        value = row.get(column)
        if pd.notna(value):
            parts.append(f"{label} {float(value):.4f}")
    return " / ".join(parts)


def _rows_by_id(frame: pd.DataFrame, finding_ids: list[str]) -> pd.DataFrame:
    rows = []
    indexed = frame.set_index("finding_id", drop=False)
    for finding_id in finding_ids:
        if finding_id not in indexed.index:
            raise ValueError(f"Missing finding_id={finding_id!r}")
        rows.append(indexed.loc[finding_id])
    return pd.DataFrame(rows)


def _local_frontier(core: pd.DataFrame) -> pd.DataFrame:
    rows = _rows_by_id(core, LOCAL_FRONTIER_IDS)
    return pd.DataFrame(
        {
            "finding_id": rows["finding_id"],
            "method": rows["method"],
            "setting": rows["setting"],
            "metrics": [_format_metric(row) for _index, row in rows.iterrows()],
            "claim_status": [
                "local_same_domain",
                "local_transfer_frontier",
                "local_triage_frontier",
                "local_reverse_frontier",
                "local_reverse_clean_anchor",
                "local_robustness_stress",
                "local_robustness_stress",
                "local_robustness_stress",
            ],
        }
    )


def _ntire_gap_rows(core: pd.DataFrame) -> pd.DataFrame:
    rows = _rows_by_id(core, NTIRE_PROXY_IDS)
    official = float(SOTA_ANCHORS[0]["official_best_score"])
    gap_rows = []
    for _index, row in rows.iterrows():
        auc = float(row["auc"])
        gap_rows.append(
            {
                "benchmark_id": "ntire_2026_robust_aigc",
                "finding_id": row["finding_id"],
                "method": row["method"],
                "proxy_setting": row["setting"],
                "proxy_auc": auc,
                "official_sota_auc": official,
                "auc_gap_to_official_sota": auc - official,
                "comparison_validity": "proxy_not_official",
            }
        )
    return pd.DataFrame(gap_rows)


def build_sota_gap(core_results_path: Path) -> tuple[str, pd.DataFrame]:
    core = pd.read_csv(core_results_path)
    anchors = pd.DataFrame(SOTA_ANCHORS)
    local = _local_frontier(core)
    gaps = _ntire_gap_rows(core)

    gap_summary = gaps.sort_values("proxy_auc", ascending=False).copy()
    gap_summary["proxy_auc"] = gap_summary["proxy_auc"].map(lambda value: f"{value:.4f}")
    gap_summary["official_sota_auc"] = gap_summary["official_sota_auc"].map(lambda value: f"{value:.4f}")
    gap_summary["auc_gap_to_official_sota"] = gap_summary["auc_gap_to_official_sota"].map(
        lambda value: f"{value:.4f}"
    )

    local_best_transfer = core.loc[core["finding_id"].eq("ishu_to_ms_clip_standalone")].iloc[0]
    local_best_reverse = core.loc[core["finding_id"].eq("ms_to_ishu_tuned_fusion_native_tiling_best")].iloc[0]
    local_best_stress = core.loc[core["finding_id"].eq("ms_to_ishu_tuned_fusion_noise3")].iloc[0]

    lines = [
        "# SOTA Gap Report",
        "",
        f"Run date: {RUN_DATE}",
        "",
        "Generated by `scripts/build_sota_gap_report.py` from checked-in publication metrics plus manually curated official benchmark anchors.",
        "",
        "This report is intentionally conservative: it separates local proxy evidence from official benchmark performance so paper drafts do not imply leaderboard placement.",
        "",
        "## Verdict",
        "",
        "- The project is not SOTA on official public benchmarks because it has no official NTIRE 2026, ImageCLEF 2026, GenImage, or Chameleon submission/run.",
        f"- The strongest local cross-domain ranker is `{local_best_transfer['method']}` on `{local_best_transfer['setting']}` with AUC `{float(local_best_transfer['auc']):.4f}`.",
        f"- The strongest reverse-transfer operating point is `{local_best_reverse['method']}` with accuracy `{float(local_best_reverse['accuracy']):.4f}` and AUC `{float(local_best_reverse['auc']):.4f}`.",
        f"- The best local robustness-stress AUC row is `{local_best_stress['method']}` on `{local_best_stress['setting']}` with AUC `{float(local_best_stress['auc']):.4f}`, but this is still a proxy result.",
        "",
        "## External SOTA Anchors",
        "",
        _markdown_table(
            anchors,
            [
                "benchmark_id",
                "status",
                "official_metric",
                "official_best_method",
                "official_best_score",
                "comparison_validity",
                "reason",
                "source_url",
            ],
        ),
        "",
        "## Local Frontier",
        "",
        _markdown_table(local, ["claim_status", "finding_id", "method", "setting", "metrics"]),
        "",
        "## NTIRE-Style Gap",
        "",
        "These rows compare the closest local proxy AUCs against the NTIRE 2026 official top Robust ROC-AUC. They are useful for prioritization, not for claiming a leaderboard rank.",
        "",
        _markdown_table(
            gap_summary,
            [
                "finding_id",
                "method",
                "proxy_auc",
                "official_sota_auc",
                "auc_gap_to_official_sota",
                "comparison_validity",
            ],
        ),
        "",
        "## Interpretation For Submissions",
        "",
        "- Use `behind official SOTA but methodologically aligned with current robustness benchmarks` as the public framing.",
        "- The publishable contribution is the source-heldout diagnostic protocol and physics/foundation/conventional ablation story, not a leaderboard claim.",
        "- To make a SOTA claim, the next required step is running official or released benchmark splits for NTIRE-style, GenImage, Chameleon, or the next ImageCLEF/NTIRE cycle.",
        "",
    ]
    return "\n".join(lines), gaps


def main() -> None:
    args = parse_args()
    text, gaps = build_sota_gap(Path(args.core_results))

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")

    csv_path = Path(args.csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    gaps.to_csv(csv_path, index=False)

    print(out_path.resolve())
    print(csv_path.resolve())


if __name__ == "__main__":
    main()
