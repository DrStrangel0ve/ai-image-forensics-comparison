from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RUN_DATE = "2026-06-14"

BENCHMARKS = [
    {
        "benchmark_id": "ntire_2026_robust_aigc",
        "name": "NTIRE 2026 Robust AI-Generated Image Detection in the Wild",
        "official_status": "closed_not_submitted",
        "official_score": "none",
        "local_data_status": "no_official_validation_or_test_labels",
        "proxy_status": "proxy_tested",
        "nearest_proxy": "Defactify/MS COCOAI source-balanced transfer and robustness transforms.",
        "readiness_grade": "protocol_ready_not_scored",
        "blocker": "The 2026 challenge window is closed and no official scored validation/test labels are checked into the repo.",
        "next_action": "Monitor NTIRE 2027 or released labeled splits; run SCP-Fusion on the official train/validation protocol if labels become available.",
        "evidence_report": "reports/ms_cocoai_robustness_variants.md; reports/assets/publication_core_results.csv",
        "source_url": "https://arxiv.org/abs/2604.11487",
    },
    {
        "benchmark_id": "imageclef_2026_deepfake",
        "name": "ImageCLEF 2026 Deepfake Detection and Generation",
        "official_status": "closed_not_submitted",
        "official_score": "none",
        "local_data_status": "no_official_task_data",
        "proxy_status": "protocol_inspired_only",
        "nearest_proxy": "Source-heldout generator evaluation, triage, and failure-case analysis on Ishu/MS COCOAI.",
        "readiness_grade": "concept_ready_no_official_data",
        "blocker": "The 2026 detection submission deadline is past and no ImageCLEF task export is checked into the repo.",
        "next_action": "Watch for ImageCLEF 2027 registration; adapt the existing source-heldout and failure-grid scripts to the task format.",
        "evidence_report": "reports/source_holdout_triage_2026_06_12.md; reports/qualitative_failure_cases_2026_06_12.md",
        "source_url": "https://www.imageclef.org/2026/deepfake-detection-and-generation",
    },
]

PROXY_METRIC_IDS = {
    "ntire_2026_robust_aigc": [
        "ishu_to_ms_clip_standalone",
        "ishu_to_ms_scp_fusion_all_foundation",
        "ms_to_ishu_tuned_fusion_constraint_sweep_best",
        "ms_to_ishu_tuned_fusion_jpeg30",
        "ms_to_ishu_tuned_fusion_blur1",
        "ms_to_ishu_tuned_fusion_resize_half",
        "ms_to_ishu_tuned_fusion_noise3",
        "ms_to_ishu_tuned_fusion_social_square",
    ],
    "imageclef_2026_deepfake": [
        "ishu_to_ms_clip_standalone",
        "ishu_to_ms_triage5_clip_standalone",
        "ishu_to_ms_triage5_scp_fusion_all_foundation",
        "ms_to_ishu_source_holdout_tuned_fusion",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an official-vs-proxy readiness report for external challenge benchmarks."
    )
    parser.add_argument(
        "--core-results",
        default="reports/assets/publication_core_results.csv",
        help="Canonical publication result table.",
    )
    parser.add_argument(
        "--out-path",
        default="reports/external_benchmark_readiness_2026_06_14.md",
        help="Markdown readiness report to write.",
    )
    parser.add_argument(
        "--status-out",
        default="reports/assets/external_benchmark_readiness.csv",
        help="Machine-readable benchmark readiness table.",
    )
    parser.add_argument(
        "--metrics-out",
        default="reports/assets/external_benchmark_proxy_metrics.csv",
        help="Machine-readable proxy metric rows.",
    )
    return parser.parse_args()


def _format_metric(value: object) -> str:
    if pd.isna(value):
        return ""
    return f"{float(value):.4f}"


def _metric_summary(row: pd.Series) -> str:
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
        if column in row and pd.notna(row[column]):
            parts.append(f"{label} {_format_metric(row[column])}")
    return " / ".join(parts)


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


def _proxy_metrics(core_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    core_by_id = core_results.set_index("finding_id", drop=False)
    for benchmark_id, finding_ids in PROXY_METRIC_IDS.items():
        for finding_id in finding_ids:
            if finding_id not in core_by_id.index:
                raise ValueError(f"Missing proxy finding_id={finding_id!r}")
            source = core_by_id.loc[finding_id]
            rows.append(
                {
                    "benchmark_id": benchmark_id,
                    "finding_id": finding_id,
                    "setting": source["setting"],
                    "method": source["method"],
                    "metric_summary": _metric_summary(source),
                    "accuracy": source.get("accuracy", pd.NA),
                    "auc": source.get("auc", pd.NA),
                    "brier": source.get("brier", pd.NA),
                    "ece": source.get("ece", pd.NA),
                    "predicted_fake_rate": source.get("predicted_fake_rate", pd.NA),
                    "coverage": source.get("coverage", pd.NA),
                    "decided_accuracy": source.get("decided_accuracy", pd.NA),
                    "source": source["source"],
                }
            )
    return pd.DataFrame(rows)


def build_readiness(core_results_path: Path) -> tuple[str, pd.DataFrame, pd.DataFrame]:
    core_results = pd.read_csv(core_results_path)
    status = pd.DataFrame(BENCHMARKS)
    metrics = _proxy_metrics(core_results)

    summary = status.groupby(["official_status", "proxy_status"], sort=False).size().reset_index(name="count")
    lines = [
        "# External Benchmark Readiness",
        "",
        f"Run date: {RUN_DATE}",
        "",
        "Generated by `scripts/build_external_benchmark_readiness.py` from checked-in external benchmark metadata and canonical publication metrics.",
        "",
        "This report prevents an easy overclaim: NTIRE 2026 and ImageCLEF 2026 are tracked as closed external benchmark targets, not as official scored submissions.",
        "",
        "## Summary",
        "",
        _markdown_table(summary, ["official_status", "proxy_status", "count"]),
        "",
        "## Official Status",
        "",
        _markdown_table(
            status,
            [
                "benchmark_id",
                "official_status",
                "official_score",
                "local_data_status",
                "proxy_status",
                "readiness_grade",
                "blocker",
                "next_action",
            ],
        ),
        "",
        "## Proxy Evidence",
        "",
        _markdown_table(
            metrics,
            ["benchmark_id", "finding_id", "method", "setting", "metric_summary", "source"],
        ),
        "",
        "## Interpretation",
        "",
        "- NTIRE readiness is strongest at the protocol level: the repo already has source-balanced, transform-stressed proxy evidence, but no official challenge score.",
        "- ImageCLEF readiness is mostly methodological: the repo has generator-heldout triage and qualitative failure analysis, but no local ImageCLEF task data.",
        "- Any poster or paper should say `NTIRE/ImageCLEF-style` or `proxy robustness evidence`; it should not imply official participation or leaderboard placement.",
        "",
    ]
    return "\n".join(lines), status, metrics


def main() -> None:
    args = parse_args()
    text, status, metrics = build_readiness(Path(args.core_results))

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")

    status_path = Path(args.status_out)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status.to_csv(status_path, index=False)

    metrics_path = Path(args.metrics_out)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(metrics_path, index=False)

    print(out_path.resolve())
    print(status_path.resolve())
    print(metrics_path.resolve())


if __name__ == "__main__":
    main()
