from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RUN_DATE = "2026-06-14"

SCENARIOS = [
    {
        "scenario_id": "same_domain_ishu",
        "title": "Ishu Same-Domain Anchor",
        "finding_ids": [
            "ishu_same_combined_v3",
            "ishu_same_resnet18",
            "ishu_same_physics_guided",
            "ishu_same_combined_v4_raw",
            "ishu_same_combined_v4_selectk60",
        ],
        "criteria": ["best_accuracy", "best_auc"],
        "caveat": "Same-domain wins are useful anchors, not deployment evidence under generator shift.",
    },
    {
        "scenario_id": "forward_transfer_ishu_to_ms",
        "title": "Ishu -> MS COCOAI Transfer",
        "finding_ids": [
            "ishu_to_ms_combined_v3",
            "ishu_to_ms_resnet18",
            "ishu_to_ms_physics_guided",
            "ishu_to_ms_convnext_tiny_frozen",
            "ishu_to_ms_combined_v4_raw",
            "ishu_to_ms_combined_v4_selectk60",
            "ishu_to_ms_scp_fusion_v0",
            "ishu_to_ms_scp_fusion_dinov2",
            "ishu_to_ms_scp_fusion_clip",
            "ishu_to_ms_scp_fusion_all_foundation",
            "ishu_to_ms_clip_standalone",
        ],
        "criteria": ["best_accuracy", "best_auc", "best_brier", "best_ece", "fake_rate_closest_0p5"],
        "caveat": "Forward transfer separates ranking strength from calibrated/default binary decisions.",
    },
    {
        "scenario_id": "forward_triage_ishu_to_ms",
        "title": "Ishu -> MS COCOAI Source-Heldout Triage",
        "finding_ids": [
            "ishu_to_ms_triage5_scp_fusion_all_foundation",
            "ishu_to_ms_triage5_clip_standalone",
        ],
        "criteria": ["best_decided_accuracy", "best_coverage"],
        "caveat": "Triage coverage is intentionally partial; it is not full automation.",
    },
    {
        "scenario_id": "reverse_transfer_ms_to_ishu",
        "title": "MS COCOAI -> Ishu Reverse Transfer",
        "finding_ids": [
            "ms_to_ishu_physics_guided",
            "ms_to_ishu_clip_vit_b_32",
            "ms_to_ishu_score_fusion_all6_temp_balanced",
            "ms_to_ishu_branch_dropout_auc",
            "ms_to_ishu_source_cap_accuracy",
            "ms_to_ishu_source_utility_cap_0p48",
            "ms_to_ishu_source_holdout_mean_utility_cap_0p48",
            "ms_to_ishu_source_holdout_tuned_fusion",
            "ms_to_ishu_tuned_fusion_constraint_sweep_best",
            "ms_to_ishu_tuned_fusion_native_tiling_best",
        ],
        "criteria": ["best_accuracy", "best_auc", "best_brier", "best_ece", "fake_rate_closest_0p5"],
        "caveat": "Reverse-transfer winners depend strongly on whether ranking, calibration, or fake-call bias is prioritized.",
    },
]

CRITERIA = {
    "best_accuracy": {"metric": "accuracy", "label": "Best accuracy", "maximize": True},
    "best_auc": {"metric": "auc", "label": "Best AUC", "maximize": True},
    "best_brier": {"metric": "brier", "label": "Best Brier", "maximize": False},
    "best_ece": {"metric": "ece", "label": "Best ECE", "maximize": False},
    "best_decided_accuracy": {"metric": "decided_accuracy", "label": "Best decided accuracy", "maximize": True},
    "best_coverage": {"metric": "coverage", "label": "Best triage coverage", "maximize": True},
    "fake_rate_closest_0p5": {
        "metric": "fake_call_rate_gap",
        "source_metric": "predicted_fake_rate",
        "label": "Smallest fake-call gap from 0.5",
        "maximize": False,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a compact method-family comparison report from publication_core_results.csv."
    )
    parser.add_argument(
        "--core-results",
        default="reports/assets/publication_core_results.csv",
        help="Canonical publication result table.",
    )
    parser.add_argument(
        "--csv-out",
        default="reports/assets/method_family_comparison.csv",
        help="Machine-readable method-family comparison CSV.",
    )
    parser.add_argument(
        "--report-out",
        default="reports/method_family_comparison_2026_06_14.md",
        help="Markdown report summarizing method-family winners by scenario and criterion.",
    )
    return parser.parse_args()


def _method_family(method: object) -> str:
    text = str(method).lower()
    if "scp-fusion" in text or "fusion" in text or "source-" in text or "source " in text:
        return "score/source fusion"
    if "clip" in text or "dinov2" in text or "convnext" in text:
        return "frozen foundation"
    if "physics-guided" in text:
        return "physics-guided neural"
    if "resnet" in text:
        return "task-trained neural"
    if "combined_v" in text:
        return "physical/signal conventional"
    return "other"


def _scenario_frame(core: pd.DataFrame, finding_ids: list[str]) -> pd.DataFrame:
    rows = []
    for finding_id in finding_ids:
        matches = core[core["finding_id"] == finding_id]
        if matches.empty:
            raise ValueError(f"Missing finding_id={finding_id!r}")
        rows.append(matches.iloc[0].to_dict())
    frame = pd.DataFrame(rows)
    frame["method_family"] = [_method_family(method) for method in frame["method"]]
    if "predicted_fake_rate" in frame.columns:
        frame["fake_call_rate_gap"] = (frame["predicted_fake_rate"].astype(float) - 0.5).abs()
    return frame


def _metric_value(row: pd.Series, metric: str) -> float | None:
    if metric not in row or pd.isna(row[metric]):
        return None
    return float(row[metric])


def _winner_row(frame: pd.DataFrame, criterion_id: str) -> dict[str, object] | None:
    spec = CRITERIA[criterion_id]
    metric = str(spec["metric"])
    candidates = frame[frame[metric].notna()].copy()
    if candidates.empty:
        return None
    candidates["_rank_metric"] = candidates[metric].astype(float)
    candidates = candidates.sort_values("_rank_metric", ascending=not bool(spec["maximize"]), kind="mergesort")
    winner = candidates.iloc[0]
    runner = candidates.iloc[1] if len(candidates) > 1 else None
    winner_value = float(winner["_rank_metric"])
    if runner is None:
        runner_margin = None
        runner_method = ""
        runner_family = ""
        runner_finding_id = ""
    else:
        runner_value = float(runner["_rank_metric"])
        runner_margin = winner_value - runner_value if spec["maximize"] else runner_value - winner_value
        runner_method = str(runner["method"])
        runner_family = str(runner["method_family"])
        runner_finding_id = str(runner["finding_id"])
    return {
        "criterion": criterion_id,
        "criterion_label": spec["label"],
        "metric": metric,
        "metric_value": winner_value,
        "runner_up_margin": runner_margin,
        "winner_family": winner["method_family"],
        "finding_id": winner["finding_id"],
        "method": winner["method"],
        "accuracy": _metric_value(winner, "accuracy"),
        "auc": _metric_value(winner, "auc"),
        "brier": _metric_value(winner, "brier"),
        "ece": _metric_value(winner, "ece"),
        "fake_call_rate": _metric_value(winner, "predicted_fake_rate"),
        "coverage": _metric_value(winner, "coverage"),
        "decided_accuracy": _metric_value(winner, "decided_accuracy"),
        "runner_up_finding_id": runner_finding_id,
        "runner_up_family": runner_family,
        "runner_up_method": runner_method,
    }


def build_method_family_comparison(core_results: Path) -> pd.DataFrame:
    core = pd.read_csv(core_results)
    rows = []
    for scenario in SCENARIOS:
        frame = _scenario_frame(core, scenario["finding_ids"])
        for criterion_id in scenario["criteria"]:
            winner = _winner_row(frame, criterion_id)
            if winner is None:
                continue
            rows.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "scenario": scenario["title"],
                    **winner,
                    "scenario_caveat": scenario["caveat"],
                }
            )
    return pd.DataFrame(rows)


def _format_value(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


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
        lines.append("| " + " | ".join(_markdown_escape(_format_value(value)) for value in row) + " |")
    return "\n".join(lines)


def write_report(frame: pd.DataFrame, report_out: Path) -> None:
    family_counts = (
        frame.groupby("winner_family", sort=False)
        .size()
        .reset_index(name="winner_count")
        .sort_values(["winner_count", "winner_family"], ascending=[False, True])
    )
    display_columns = [
        "scenario",
        "criterion_label",
        "winner_family",
        "method",
        "metric_value",
        "runner_up_margin",
        "runner_up_method",
        "scenario_caveat",
    ]
    lines = [
        "# Method Family Comparison",
        "",
        f"Run date: {RUN_DATE}",
        "",
        "Generated by `scripts/build_method_family_comparison.py` from `reports/assets/publication_core_results.csv`.",
        "",
        "This report is a paper-facing decision aid. It names which method family wins each local scenario/criterion, while keeping ranking, calibration, fake-call bias, and triage coverage separate.",
        "",
        "## Winner Counts",
        "",
        _markdown_table(family_counts, list(family_counts.columns)),
        "",
        "## Scenario Winners",
        "",
        _markdown_table(frame, display_columns),
        "",
        "## Editorial Use",
        "",
        "- Use this table to keep the WIFS/DFF narrative multi-objective: CLIP is the forward-transfer ranker, physics-guided fusion is the same-domain anchor, and source-aware fusion is the stronger reverse operating-point family.",
        "- Do not collapse these rows into a single overall winner; the criteria deliberately measure different forensic behaviors.",
        "",
    ]
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    frame = build_method_family_comparison(Path(args.core_results))
    csv_path = Path(args.csv_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(csv_path, index=False)
    write_report(frame, Path(args.report_out))
    print(Path(args.report_out).resolve())
    print(csv_path.resolve())


if __name__ == "__main__":
    main()
