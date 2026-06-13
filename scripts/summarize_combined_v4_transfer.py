from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_RUNS = [
    "combined_v3_logreg",
    "combined_v4_logreg",
    "combined_v4_logreg_selectk60",
]

METRIC_COLUMNS = [
    "accuracy",
    "roc_auc",
    "brier_score",
    "expected_calibration_error",
    "precision",
    "recall",
    "f1",
    "fake_call_rate",
    "real_false_positive_rate",
    "fake_miss_rate",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize combined_v4 same-domain and transfer metrics from runs/."
    )
    parser.add_argument("--source-root", default="runs/combined_v4_full_transfer")
    parser.add_argument("--transfer-root", default="runs/combined_v4_full_transfer_to_ms")
    parser.add_argument("--run", action="append", dest="runs", default=None)
    parser.add_argument(
        "--seed-out",
        default="reports/assets/combined_v4_full_transfer_seed_summary.csv",
    )
    parser.add_argument(
        "--mean-out",
        default="reports/assets/combined_v4_full_transfer_mean_summary.csv",
    )
    parser.add_argument(
        "--delta-out",
        default="reports/assets/combined_v4_full_transfer_delta_summary.csv",
    )
    parser.add_argument(
        "--report-out",
        default="reports/combined_v4_full_transfer_summary_2026_06_13.md",
    )
    parser.add_argument("--run-date", default="2026-06-13")
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_from_path(path: Path) -> int:
    for part in path.parts:
        if part.startswith("seed") and part[4:].isdigit():
            return int(part[4:])
    raise ValueError(f"Could not infer seed from {path}")


def _select_k(run: str, metrics: dict[str, Any]) -> int:
    for key in ["effective_select_k", "select_k"]:
        value = metrics.get(key)
        if value is not None:
            return int(value)
    if "selectk" in run:
        suffix = run.rsplit("selectk", maxsplit=1)[-1]
        if suffix.isdigit():
            return int(suffix)
    return 0


def _confusion_rates(metrics: dict[str, Any]) -> dict[str, float | int | None]:
    matrix = metrics.get("confusion_matrix")
    if not matrix:
        return {
            "support_real": None,
            "support_fake": None,
            "fake_call_rate": None,
            "real_false_positive_rate": None,
            "fake_miss_rate": None,
        }
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    total = tn + fp + fn + tp
    real_total = tn + fp
    fake_total = fn + tp
    return {
        "support_real": real_total,
        "support_fake": fake_total,
        "fake_call_rate": (fp + tp) / total if total else None,
        "real_false_positive_rate": fp / real_total if real_total else None,
        "fake_miss_rate": fn / fake_total if fake_total else None,
    }


def _phase_paths(root: Path, runs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for run in runs:
        paths.extend(sorted(root.glob(f"seed*/{run}/metrics.json")))
    return paths


def _rows_for_phase(
    root: Path,
    phase: str,
    phase_label: str,
    runs: list[str],
) -> list[dict[str, Any]]:
    rows = []
    for path in _phase_paths(root, runs):
        metrics = _read_json(path)
        run = path.parent.name
        rates = _confusion_rates(metrics)
        rows.append(
            {
                "phase": phase,
                "phase_label": phase_label,
                "seed": _seed_from_path(path),
                "run": run,
                "feature_set": metrics.get("feature_set"),
                "classifier": metrics.get("classifier"),
                "select_k": _select_k(run, metrics),
                "method": metrics.get("method"),
                "accuracy": metrics.get("accuracy"),
                "roc_auc": metrics.get("roc_auc"),
                "brier_score": metrics.get("brier_score"),
                "expected_calibration_error": metrics.get("expected_calibration_error"),
                "precision": metrics.get("precision"),
                "recall": metrics.get("recall"),
                "f1": metrics.get("f1"),
                "threshold": metrics.get("threshold"),
                "n_train": metrics.get("n_train"),
                "n_test": metrics.get("n_test"),
                "n_target": metrics.get("n_target"),
                **rates,
                "metrics_path": path.as_posix(),
            }
        )
    return rows


def load_seed_summary(source_root: Path, transfer_root: Path, runs: list[str]) -> pd.DataFrame:
    rows = []
    rows.extend(
        _rows_for_phase(
            source_root,
            "ishu_holdout",
            "Ishu holdout split",
            runs,
        )
    )
    rows.extend(
        _rows_for_phase(
            transfer_root,
            "ishu_to_ms_cocoai",
            "Ishu -> source-balanced MS COCOAI",
            runs,
        )
    )
    if not rows:
        raise FileNotFoundError(
            f"No metrics found under {source_root} or {transfer_root} for runs {runs}"
        )
    return pd.DataFrame(rows).sort_values(["phase", "seed", "run"]).reset_index(drop=True)


def _mean_ci(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return pd.Series({"mean": None, "std": None, "min": None, "max": None})
    return pd.Series(
        {
            "mean": numeric.mean(),
            "std": numeric.std(ddof=1) if len(numeric) > 1 else 0.0,
            "min": numeric.min(),
            "max": numeric.max(),
        }
    )


def build_mean_summary(seed_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["phase", "phase_label", "run", "feature_set", "classifier", "select_k"]
    for keys, group in seed_summary.groupby(group_cols, sort=False, dropna=False):
        row = dict(zip(group_cols, keys))
        row["n_seeds"] = int(group["seed"].nunique())
        row["seeds"] = ",".join(str(seed) for seed in sorted(group["seed"].unique()))
        for column in METRIC_COLUMNS:
            stats = _mean_ci(group[column])
            row[f"{column}_mean"] = stats["mean"]
            row[f"{column}_std"] = stats["std"]
            row[f"{column}_min"] = stats["min"]
            row[f"{column}_max"] = stats["max"]
        rows.append(row)
    return pd.DataFrame(rows)


def build_delta_summary(seed_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    baseline_name = "combined_v3_logreg"
    candidates = [run for run in seed_summary["run"].unique() if run != baseline_name]
    for phase, phase_group in seed_summary.groupby("phase", sort=False):
        phase_label = str(phase_group["phase_label"].iloc[0])
        for candidate in candidates:
            baseline = phase_group[phase_group["run"] == baseline_name].set_index("seed")
            cand = phase_group[phase_group["run"] == candidate].set_index("seed")
            common = sorted(set(baseline.index).intersection(set(cand.index)))
            if not common:
                continue
            row: dict[str, Any] = {
                "phase": phase,
                "phase_label": phase_label,
                "candidate": candidate,
                "baseline": baseline_name,
                "n_paired_seeds": len(common),
                "seeds": ",".join(str(seed) for seed in common),
            }
            for column in METRIC_COLUMNS:
                diffs = cand.loc[common, column].astype(float) - baseline.loc[common, column].astype(float)
                row[f"{column}_delta_mean"] = diffs.mean()
                row[f"{column}_delta_min"] = diffs.min()
                row[f"{column}_delta_max"] = diffs.max()
            rows.append(row)
    return pd.DataFrame(rows)


def _format(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _index, row in frame.iterrows():
        lines.append("| " + " | ".join(_format(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def build_report(
    seed_summary: pd.DataFrame,
    mean_summary: pd.DataFrame,
    delta_summary: pd.DataFrame,
    run_date: str,
) -> str:
    seeds = sorted(seed_summary["seed"].unique())
    seed_label = ", ".join(str(seed) for seed in seeds)
    has_full_three_seed = len(seeds) >= 3
    status = (
        "This is now a repeated-seed transfer summary."
        if has_full_three_seed
        else "This is a preliminary seed slice, not a promotion decision."
    )

    lines = [
        "# combined_v4 Full-Transfer Summary",
        "",
        f"Run date: {run_date}",
        "",
        (
            "This report summarizes the checked local `metrics.json` files for the "
            "`combined_v4` transfer gate: Ishu same-domain holdout plus Ishu -> "
            "source-balanced MS COCOAI transfer."
        ),
        "",
        f"Seeds included: {seed_label}. {status}",
        "",
        "## Seed Results",
        "",
        _markdown_table(
            seed_summary,
            [
                "phase_label",
                "seed",
                "run",
                "accuracy",
                "roc_auc",
                "brier_score",
                "expected_calibration_error",
                "fake_call_rate",
                "real_false_positive_rate",
                "fake_miss_rate",
            ],
        ),
        "",
        "## Mean Results",
        "",
        _markdown_table(
            mean_summary,
            [
                "phase_label",
                "run",
                "n_seeds",
                "accuracy_mean",
                "roc_auc_mean",
                "brier_score_mean",
                "expected_calibration_error_mean",
                "fake_call_rate_mean",
            ],
        ),
        "",
        "## Paired Delta Versus combined_v3",
        "",
        _markdown_table(
            delta_summary,
            [
                "phase_label",
                "candidate",
                "n_paired_seeds",
                "accuracy_delta_mean",
                "roc_auc_delta_mean",
                "brier_score_delta_mean",
                "expected_calibration_error_delta_mean",
                "fake_call_rate_delta_mean",
            ],
        ),
        "",
        "## Interpretation",
        "",
        (
            "For the current seed slice, `combined_v4_selectk60` is the most interesting "
            "transfer candidate because it improves MS COCOAI transfer accuracy, AUC, "
            "Brier score, and ECE versus `combined_v3`. The same model is slightly weaker "
            "on the Ishu holdout split, so it should stay an ablation until seeds 17 and 29 "
            "confirm whether this is a real cross-domain gain."
        ),
        "",
        (
            "Raw `combined_v4` remains a useful diagnostic branch: it nudges same-domain "
            "Ishu ranking upward in this seed, but it does not improve transfer AUC yet."
        ),
        "",
        "Next step: run the remaining rows in `reports/assets/combined_v4_transfer_command_manifest.csv` and regenerate this report.",
        "",
    ]
    return "\n".join(lines)


def write_outputs(
    seed_summary: pd.DataFrame,
    mean_summary: pd.DataFrame,
    delta_summary: pd.DataFrame,
    report: str,
    seed_out: Path,
    mean_out: Path,
    delta_out: Path,
    report_out: Path,
) -> None:
    for path in [seed_out, mean_out, delta_out, report_out]:
        path.parent.mkdir(parents=True, exist_ok=True)
    seed_summary.to_csv(seed_out, index=False)
    mean_summary.to_csv(mean_out, index=False)
    delta_summary.to_csv(delta_out, index=False)
    report_out.write_text(report, encoding="utf-8")


def main() -> None:
    args = parse_args()
    runs = args.runs or DEFAULT_RUNS
    seed_summary = load_seed_summary(Path(args.source_root), Path(args.transfer_root), runs)
    mean_summary = build_mean_summary(seed_summary)
    delta_summary = build_delta_summary(seed_summary)
    report = build_report(seed_summary, mean_summary, delta_summary, args.run_date)
    write_outputs(
        seed_summary,
        mean_summary,
        delta_summary,
        report,
        Path(args.seed_out),
        Path(args.mean_out),
        Path(args.delta_out),
        Path(args.report_out),
    )
    print(Path(args.report_out).resolve())
    print(Path(args.seed_out).resolve())
    print(Path(args.mean_out).resolve())
    print(Path(args.delta_out).resolve())


if __name__ == "__main__":
    main()
