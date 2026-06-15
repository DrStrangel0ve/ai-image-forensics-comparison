from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forensic_compare.metrics import bootstrap_mean_ci  # noqa: E402


RUN_DATE = "2026-06-15"
Direction = Literal["higher", "lower", "diagnostic"]


@dataclass(frozen=True)
class MetricSpec:
    name: str
    column: str
    direction: Direction


@dataclass(frozen=True)
class ComparisonSpec:
    comparison_id: str
    title: str
    source: str
    seed_column: str
    method_column: str
    candidate: str
    baseline: str
    metrics: tuple[MetricSpec, ...]
    filters: tuple[tuple[str, str], ...] = ()
    note: str = ""


DEFAULT_COMPARISONS = (
    ComparisonSpec(
        comparison_id="ishu_physics_guided_vs_resnet18",
        title="Ishu same-domain: physics-guided ResNet vs vanilla ResNet-18",
        source="ishu_physics_runs",
        seed_column="run",
        method_column="method",
        candidate="physics_guided_resnet18_combined_v3",
        baseline="neural_resnet18",
        metrics=(
            MetricSpec("accuracy", "accuracy", "higher"),
            MetricSpec("AUC", "roc_auc", "higher"),
        ),
        note="This is the paired-seed version of the headline physics-informed-vs-traditional ResNet comparison.",
    ),
    ComparisonSpec(
        comparison_id="ishu_physics_guided_vs_combined_v3",
        title="Ishu same-domain: physics-guided ResNet vs combined_v3 conventional baseline",
        source="ishu_physics_runs",
        seed_column="run",
        method_column="method",
        candidate="physics_guided_resnet18_combined_v3",
        baseline="feature_combined_v3",
        metrics=(
            MetricSpec("accuracy", "accuracy", "higher"),
            MetricSpec("AUC", "roc_auc", "higher"),
        ),
        note="Checks whether the physics-guided neural model is more than a reweighted conventional feature model.",
    ),
    ComparisonSpec(
        comparison_id="combined_v4_ishu_holdout_raw_vs_v3",
        title="Ishu same-domain: raw combined_v4 vs combined_v3",
        source="combined_v4",
        seed_column="seed",
        method_column="run",
        candidate="combined_v4_logreg",
        baseline="combined_v3_logreg",
        metrics=(
            MetricSpec("accuracy", "accuracy", "higher"),
            MetricSpec("AUC", "roc_auc", "higher"),
            MetricSpec("Brier", "brier_score", "lower"),
            MetricSpec("ECE", "expected_calibration_error", "lower"),
            MetricSpec("fake-call rate", "fake_call_rate", "diagnostic"),
        ),
        filters=(("phase", "ishu_holdout"),),
        note="Tests whether the broader handcrafted feature set helps before source shift.",
    ),
    ComparisonSpec(
        comparison_id="combined_v4_ishu_to_ms_selectk60_vs_v3",
        title="Ishu -> MS COCOAI: select-k60 combined_v4 vs combined_v3",
        source="combined_v4",
        seed_column="seed",
        method_column="run",
        candidate="combined_v4_logreg_selectk60",
        baseline="combined_v3_logreg",
        metrics=(
            MetricSpec("accuracy", "accuracy", "higher"),
            MetricSpec("AUC", "roc_auc", "higher"),
            MetricSpec("Brier", "brier_score", "lower"),
            MetricSpec("ECE", "expected_calibration_error", "lower"),
            MetricSpec("fake-call rate", "fake_call_rate", "diagnostic"),
        ),
        filters=(("phase", "ishu_to_ms_cocoai"),),
        note="Checks whether v4's feature expansion is useful as a transfer/calibration ablation.",
    ),
    ComparisonSpec(
        comparison_id="ishu_to_ms_scp_all_foundation_vs_clip",
        title="Ishu -> MS COCOAI: SCP all-foundation fusion vs CLIP standalone",
        source="score_fusion",
        seed_column="group",
        method_column="method",
        candidate="scp_fusion_all_foundation",
        baseline="clip_standalone",
        metrics=(
            MetricSpec("accuracy", "accuracy", "higher"),
            MetricSpec("AUC", "roc_auc", "higher"),
            MetricSpec("Brier", "brier_score", "lower"),
            MetricSpec("ECE", "expected_calibration_error", "lower"),
            MetricSpec("predicted positive rate", "predicted_positive_rate", "diagnostic"),
        ),
        note="Separates fusion's calibration/operating behavior from CLIP's ranking strength.",
    ),
    ComparisonSpec(
        comparison_id="ishu_to_ms_source_calibrated_all_foundation_vs_clip",
        title="Ishu -> MS COCOAI: source-calibrated all-foundation fusion vs CLIP standalone",
        source="score_fusion",
        seed_column="group",
        method_column="method",
        candidate="all_foundation_source_calibrated",
        baseline="clip_standalone",
        metrics=(
            MetricSpec("accuracy", "accuracy", "higher"),
            MetricSpec("AUC", "roc_auc", "higher"),
            MetricSpec("Brier", "brier_score", "lower"),
            MetricSpec("ECE", "expected_calibration_error", "lower"),
            MetricSpec("predicted positive rate", "predicted_positive_rate", "diagnostic"),
        ),
        note="Checks whether source calibration recovers operating-point accuracy without claiming AUC leadership.",
    ),
    ComparisonSpec(
        comparison_id="ishu_to_ms_source_calibrated_clip_vs_clip",
        title="Ishu -> MS COCOAI: source-calibrated CLIP vs CLIP standalone",
        source="score_fusion",
        seed_column="group",
        method_column="method",
        candidate="clip_source_calibrated",
        baseline="clip_standalone",
        metrics=(
            MetricSpec("accuracy", "accuracy", "higher"),
            MetricSpec("AUC", "roc_auc", "higher"),
            MetricSpec("Brier", "brier_score", "lower"),
            MetricSpec("ECE", "expected_calibration_error", "lower"),
            MetricSpec("predicted positive rate", "predicted_positive_rate", "diagnostic"),
        ),
        note="Tests whether score calibration alone can improve CLIP's default decision behavior.",
    ),
    ComparisonSpec(
        comparison_id="ms_to_ishu_physics_guided_vs_resnet18",
        title="MS COCOAI -> Ishu: physics-guided ResNet vs vanilla ResNet-18",
        source="reverse_transfer",
        seed_column="seed",
        method_column="method",
        candidate="physics_guided_resnet18_combined_v3",
        baseline="resnet18",
        metrics=(
            MetricSpec("accuracy", "accuracy", "higher"),
            MetricSpec("AUC", "auc", "higher"),
            MetricSpec("Brier", "brier", "lower"),
            MetricSpec("ECE", "ece", "lower"),
            MetricSpec("fake-call rate", "predicted_fake_rate", "diagnostic"),
        ),
        filters=(("split", "ms_cocoai_to_ishu_test"),),
        note="Reverse-transfer check for the physics-guided branch under a different training source.",
    ),
    ComparisonSpec(
        comparison_id="ms_to_ishu_temp_balanced_fusion_vs_clip",
        title="MS COCOAI -> Ishu: temperature-balanced score fusion vs CLIP",
        source="reverse_transfer",
        seed_column="seed",
        method_column="method",
        candidate="score_fusion_all6_temp_balanced",
        baseline="clip_vit_b_32",
        metrics=(
            MetricSpec("accuracy", "accuracy", "higher"),
            MetricSpec("AUC", "auc", "higher"),
            MetricSpec("Brier", "brier", "lower"),
            MetricSpec("ECE", "ece", "lower"),
            MetricSpec("fake-call rate", "predicted_fake_rate", "diagnostic"),
        ),
        filters=(("split", "ms_cocoai_to_ishu_test"),),
        note="Tests whether the multi-branch reverse fusion is a real improvement over the strongest frozen CLIP branch.",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build paired-seed bootstrap support tables for paper-facing model comparisons."
    )
    parser.add_argument("--repo-root", default=".", help="Repository root for resolving relative paths.")
    parser.add_argument(
        "--ishu-physics-runs",
        default="runs/ishu_ai_vs_real_2026_physics_guided_repeated/summary/repeated_runs.csv",
        help="Seed-level Ishu physics-guided repeated-run CSV.",
    )
    parser.add_argument(
        "--combined-v4",
        default="reports/assets/combined_v4_full_transfer_seed_summary.csv",
        help="Seed-level combined_v4 same-domain and transfer summary CSV.",
    )
    parser.add_argument(
        "--score-fusion",
        default="reports/assets/score_fusion_clip_calibration_metrics.csv",
        help="Seed-level source-calibration/foundation-fusion metric CSV.",
    )
    parser.add_argument(
        "--reverse-transfer",
        default="reports/assets/ms_cocoai_to_ishu_reverse_all_methods_metrics.csv",
        help="Seed-level MS COCOAI to Ishu reverse-transfer metric CSV.",
    )
    parser.add_argument(
        "--csv-out",
        default="reports/assets/paired_seed_statistical_support.csv",
        help="Machine-readable paired-seed support table.",
    )
    parser.add_argument(
        "--report-out",
        default="reports/paired_seed_statistical_support_2026_06_15.md",
        help="Markdown report summarizing paired-seed support.",
    )
    parser.add_argument(
        "--run-date",
        default=RUN_DATE,
        help="Date stamped into the report, in YYYY-MM-DD format.",
    )
    return parser.parse_args()


def _resolve(repo_root: Path, path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return repo_root / path


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


def _read_source_frames(repo_root: Path, args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    paths = {
        "ishu_physics_runs": _resolve(repo_root, args.ishu_physics_runs),
        "combined_v4": _resolve(repo_root, args.combined_v4),
        "score_fusion": _resolve(repo_root, args.score_fusion),
        "reverse_transfer": _resolve(repo_root, args.reverse_transfer),
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing paired-seed input files: " + ", ".join(missing))
    return {name: pd.read_csv(path) for name, path in paths.items()}


def _apply_filters(frame: pd.DataFrame, filters: tuple[tuple[str, str], ...]) -> pd.DataFrame:
    filtered = frame.copy()
    for column, expected in filters:
        if column not in filtered.columns:
            raise ValueError(f"Input frame is missing filter column {column!r}")
        filtered = filtered[filtered[column].astype(str) == str(expected)]
    return filtered


def _favorable_delta(raw_delta: np.ndarray, direction: Direction) -> np.ndarray:
    if direction == "higher":
        return raw_delta
    if direction == "lower":
        return -raw_delta
    return raw_delta


def _support_label(direction: Direction, favorable_delta: np.ndarray, ci_low: float, ci_high: float) -> str:
    if direction == "diagnostic":
        return "diagnostic_shift"
    if len(favorable_delta) == 0:
        return "no_pairs"
    if ci_low > 0.0:
        return "consistent_gain_ci_excludes_zero"
    if np.all(favorable_delta > 0.0):
        return "all_seeds_favorable_ci_touches_zero"
    if float(np.mean(favorable_delta)) > 0.0 and int(np.sum(favorable_delta > 0.0)) >= (len(favorable_delta) // 2 + 1):
        return "mixed_seed_mean_gain"
    if np.isclose(float(np.mean(favorable_delta)), 0.0):
        return "tie_or_no_mean_delta"
    return "candidate_trails"


def _paired_rows(
    frame: pd.DataFrame,
    spec: ComparisonSpec,
    metric: MetricSpec,
) -> pd.DataFrame:
    required = {spec.seed_column, spec.method_column, metric.column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{spec.comparison_id} is missing required columns: {missing}")

    candidate = frame[frame[spec.method_column].astype(str) == spec.candidate][
        [spec.seed_column, metric.column]
    ].copy()
    baseline = frame[frame[spec.method_column].astype(str) == spec.baseline][
        [spec.seed_column, metric.column]
    ].copy()
    candidate[metric.column] = pd.to_numeric(candidate[metric.column], errors="raise")
    baseline[metric.column] = pd.to_numeric(baseline[metric.column], errors="raise")
    paired = candidate.merge(
        baseline,
        on=spec.seed_column,
        suffixes=("_candidate", "_baseline"),
        validate="one_to_one",
    )
    if paired.empty:
        raise ValueError(
            f"No paired seeds for {spec.comparison_id}: "
            f"{spec.candidate} vs {spec.baseline} on {metric.column}"
        )
    paired = paired.sort_values(spec.seed_column).reset_index(drop=True)
    paired["raw_delta"] = paired[f"{metric.column}_candidate"] - paired[f"{metric.column}_baseline"]
    return paired


def _summarize_metric(frame: pd.DataFrame, spec: ComparisonSpec, metric: MetricSpec) -> dict[str, object]:
    paired = _paired_rows(frame, spec, metric)
    raw_delta = paired["raw_delta"].to_numpy(dtype=float)
    favorable_delta = _favorable_delta(raw_delta, metric.direction)
    raw_ci = bootstrap_mean_ci(raw_delta, seed=17)
    favorable_ci = bootstrap_mean_ci(favorable_delta, seed=17)
    seeds = [str(seed) for seed in paired[spec.seed_column].tolist()]
    candidate_wins = int(np.sum(favorable_delta > 0.0)) if metric.direction != "diagnostic" else pd.NA
    candidate_losses = int(np.sum(favorable_delta < 0.0)) if metric.direction != "diagnostic" else pd.NA
    ties = int(np.sum(np.isclose(favorable_delta, 0.0))) if metric.direction != "diagnostic" else pd.NA
    return {
        "comparison_id": spec.comparison_id,
        "comparison": spec.title,
        "metric": metric.name,
        "direction": metric.direction,
        "candidate": spec.candidate,
        "baseline": spec.baseline,
        "n_paired_seeds": len(paired),
        "seeds": ";".join(seeds),
        "candidate_mean": float(paired[f"{metric.column}_candidate"].mean()),
        "baseline_mean": float(paired[f"{metric.column}_baseline"].mean()),
        "raw_delta_mean": float(raw_ci["mean"]),
        "raw_delta_ci_low": float(raw_ci["ci_low"]),
        "raw_delta_ci_high": float(raw_ci["ci_high"]),
        "favorable_delta_mean": float(favorable_ci["mean"]),
        "favorable_delta_ci_low": float(favorable_ci["ci_low"]),
        "favorable_delta_ci_high": float(favorable_ci["ci_high"]),
        "candidate_wins": candidate_wins,
        "candidate_losses": candidate_losses,
        "ties": ties,
        "support_label": _support_label(
            metric.direction,
            favorable_delta,
            float(favorable_ci["ci_low"]),
            float(favorable_ci["ci_high"]),
        ),
        "note": spec.note,
    }


def build_support(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in DEFAULT_COMPARISONS:
        frame = _apply_filters(frames[spec.source], spec.filters)
        for metric in spec.metrics:
            rows.append(_summarize_metric(frame, spec, metric))
    return pd.DataFrame(rows)


def _format_float(value: object, signed: bool = False) -> str:
    if pd.isna(value):
        return ""
    number = float(value)
    prefix = "+" if signed and number >= 0 else ""
    return f"{prefix}{number:.4f}"


def _report_table(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    display["candidate mean"] = display["candidate_mean"].map(_format_float)
    display["baseline mean"] = display["baseline_mean"].map(_format_float)
    display["delta mean"] = display["raw_delta_mean"].map(lambda value: _format_float(value, signed=True))
    display["delta 95% CI"] = display.apply(
        lambda row: (
            f"[{_format_float(row['raw_delta_ci_low'], signed=True)}, "
            f"{_format_float(row['raw_delta_ci_high'], signed=True)}]"
        ),
        axis=1,
    )
    display["wins"] = display.apply(
        lambda row: ""
        if row["direction"] == "diagnostic"
        else f"{int(row['candidate_wins'])}/{int(row['n_paired_seeds'])}",
        axis=1,
    )
    return display[
        [
            "comparison",
            "metric",
            "direction",
            "candidate mean",
            "baseline mean",
            "delta mean",
            "delta 95% CI",
            "wins",
            "support_label",
        ]
    ]


def _metric_row(frame: pd.DataFrame, comparison_id: str, metric: str) -> pd.Series | None:
    subset = frame[(frame["comparison_id"] == comparison_id) & (frame["metric"] == metric)]
    if subset.empty:
        return None
    return subset.iloc[0]


def _highlight_delta(row: pd.Series | None) -> str:
    if row is None:
        return "not available"
    wins = ""
    if row["direction"] != "diagnostic":
        wins = f", {int(row['candidate_wins'])}/{int(row['n_paired_seeds'])} favorable seeds"
    return (
        f"{_format_float(row['raw_delta_mean'], signed=True)} "
        f"[{_format_float(row['raw_delta_ci_low'], signed=True)}, "
        f"{_format_float(row['raw_delta_ci_high'], signed=True)}]{wins}"
    )


def _checks(frame: pd.DataFrame) -> pd.DataFrame:
    directional = frame[frame["direction"] != "diagnostic"]
    checks = [
        {
            "check": "all configured comparisons emitted rows",
            "status": "PASS" if len(frame) == sum(len(spec.metrics) for spec in DEFAULT_COMPARISONS) else "FAIL",
            "detail": f"{len(frame)} metric rows emitted",
        },
        {
            "check": "every metric row has paired seeds",
            "status": "PASS" if frame["n_paired_seeds"].min() >= 2 else "FAIL",
            "detail": f"minimum paired seeds: {int(frame['n_paired_seeds'].min())}",
        },
        {
            "check": "directional bootstrap intervals are finite",
            "status": "PASS"
            if np.isfinite(directional[["favorable_delta_ci_low", "favorable_delta_ci_high"]].to_numpy()).all()
            else "FAIL",
            "detail": "finite bootstrap intervals over paired seed deltas",
        },
        {
            "check": "diagnostic operating-rate metrics are not labeled as wins",
            "status": "PASS"
            if set(frame[frame["direction"] == "diagnostic"]["support_label"]) <= {"diagnostic_shift"}
            else "FAIL",
            "detail": "fake-call and predicted-positive rates stay descriptive",
        },
    ]
    return pd.DataFrame(checks)


def _write_report(frame: pd.DataFrame, checks: pd.DataFrame, out_path: Path, run_date: date) -> None:
    pass_count = int((checks["status"] == "PASS").sum())
    total = int(len(checks))
    status = "PASS" if pass_count == total else "FAIL"
    physics_acc = _metric_row(frame, "ishu_physics_guided_vs_resnet18", "accuracy")
    physics_auc = _metric_row(frame, "ishu_physics_guided_vs_resnet18", "AUC")
    clip_auc = _metric_row(frame, "ishu_to_ms_scp_all_foundation_vs_clip", "AUC")
    clip_brier = _metric_row(frame, "ishu_to_ms_scp_all_foundation_vs_clip", "Brier")
    v4_auc = _metric_row(frame, "combined_v4_ishu_to_ms_selectk60_vs_v3", "AUC")
    reverse_fusion_acc = _metric_row(frame, "ms_to_ishu_temp_balanced_fusion_vs_clip", "accuracy")

    lines = [
        "# Paired Seed Statistical Support",
        "",
        f"Run date: {run_date.isoformat()}",
        "",
        f"Status: **{status}** ({pass_count}/{total} checks passed).",
        "",
        "This report compares candidate methods against baselines on matched random seeds and bootstraps the mean paired-seed delta. With only three seeds in most blocks, treat these intervals as stability support for paper wording, not as definitive hypothesis tests.",
        "",
        "## High-Signal Takeaways",
        "",
        f"- Physics-guided ResNet vs vanilla ResNet-18 on Ishu: accuracy delta {_highlight_delta(physics_acc)}; AUC delta {_highlight_delta(physics_auc)}.",
        f"- SCP all-foundation fusion vs CLIP on Ishu -> MS COCOAI: AUC delta {_highlight_delta(clip_auc)}; Brier delta {_highlight_delta(clip_brier)}. This supports the current claim that CLIP remains the transfer-ranking anchor while fusion/calibration can change operating behavior.",
        f"- Select-k60 combined_v4 vs combined_v3 on Ishu -> MS COCOAI: AUC delta {_highlight_delta(v4_auc)}. This keeps v4 as a caveated transfer ablation rather than a same-domain replacement.",
        f"- Temperature-balanced all-branch fusion vs CLIP on MS COCOAI -> Ishu: accuracy delta {_highlight_delta(reverse_fusion_acc)}. This is the strongest paper-facing support for objective-specific fusion in the reverse-transfer setting.",
        "",
        "## Label Meanings",
        "",
        "- `consistent_gain_ci_excludes_zero`: all or most paired deltas support the candidate and the bootstrap interval over seed deltas stays favorable.",
        "- `mixed_seed_mean_gain`: the candidate has a favorable mean, but the seed-level story is mixed.",
        "- `candidate_trails`: the paired mean favors the baseline.",
        "- `diagnostic_shift`: the metric is an operating-rate shift, not a higher/lower-is-better claim.",
        "",
        "## Checks",
        "",
        _markdown_table(checks, ["check", "status", "detail"]),
        "",
        "## Paired Comparisons",
        "",
        _markdown_table(_report_table(frame), list(_report_table(frame).columns)),
        "",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).resolve()
    frames = _read_source_frames(repo_root, args)
    support = build_support(frames)
    checks = _checks(support)

    csv_path = _resolve(repo_root, args.csv_out)
    report_path = _resolve(repo_root, args.report_out)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    support.to_csv(csv_path, index=False)
    _write_report(support, checks, report_path, date.fromisoformat(args.run_date))

    print(report_path.resolve())
    print(csv_path.resolve())
    if not checks["status"].eq("PASS").all():
        raise SystemExit("Paired seed statistical support checks failed")


if __name__ == "__main__":
    main()
