from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from forensic_compare.conventional import feature_names
from forensic_compare.utils import ensure_dir, read_json


METRIC_COLUMNS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "brier_score",
    "expected_calibration_error",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize conventional feature ablation metrics and selected features."
    )
    parser.add_argument("--run-root", required=True, help="Directory to recursively scan for metrics.json.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--extra-feature-base",
        default="combined_v3",
        help="Feature set treated as the baseline when counting selected extra features.",
    )
    return parser.parse_args()


def _seed_from_path(metrics_path: Path, run_root: Path) -> str:
    relative = metrics_path.relative_to(run_root)
    for part in relative.parts:
        if part.lower().startswith("seed"):
            return part[4:] or part
    return relative.parts[0] if len(relative.parts) > 1 else ""


def _run_name(metrics_path: Path) -> str:
    return metrics_path.parent.name


def _metrics_row(metrics_path: Path, run_root: Path) -> dict:
    metrics = read_json(metrics_path)
    row = {
        "seed": _seed_from_path(metrics_path, run_root),
        "run": _run_name(metrics_path),
        "method": metrics.get("method", _run_name(metrics_path)),
        "feature_set": metrics.get("feature_set"),
        "classifier": metrics.get("classifier"),
        "select_k": int(metrics.get("effective_select_k") or 0),
        "selection_score_func": metrics.get("selection_score_func"),
        "n_train": metrics.get("n_train"),
        "n_samples": metrics.get("n_samples"),
        "metrics_path": str(metrics_path),
    }
    for column in METRIC_COLUMNS:
        row[column] = metrics.get(column)
    return row


def _selected_feature_rows(metrics_path: Path, run_root: Path, base_features: set[str]) -> list[dict]:
    metrics = read_json(metrics_path)
    selected = metrics.get("selected_feature_names") or []
    seed = _seed_from_path(metrics_path, run_root)
    run = _run_name(metrics_path)
    return [
        {
            "seed": seed,
            "run": run,
            "feature": name,
            "is_extra_feature": name not in base_features,
        }
        for name in selected
    ]


def _summary_frame(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = ["run", "feature_set", "classifier", "select_k"]
    for keys, group in runs.groupby(group_columns, dropna=False, sort=True):
        row = dict(zip(group_columns, keys))
        row["n_runs"] = int(len(group))
        for metric in METRIC_COLUMNS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=0))
            row[f"{metric}_min"] = float(group[metric].min())
            row[f"{metric}_max"] = float(group[metric].max())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["roc_auc_mean", "accuracy_mean"], ascending=False)


def _selected_frequency_frame(selected: pd.DataFrame) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame(columns=["feature", "is_extra_feature", "count", "runs"])
    rows = []
    for (feature, is_extra), group in selected.groupby(["feature", "is_extra_feature"], sort=True):
        rows.append(
            {
                "feature": feature,
                "is_extra_feature": bool(is_extra),
                "count": int(len(group)),
                "runs": ",".join(sorted(group["run"].unique())),
            }
        )
    return pd.DataFrame(rows).sort_values(["count", "feature"], ascending=[False, True])


def _format_value(value) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(_format_value(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def summarize(run_root: Path, out_dir: Path, extra_feature_base: str) -> None:
    out_dir = ensure_dir(out_dir)
    metrics_paths = sorted(run_root.rglob("metrics.json"))
    if not metrics_paths:
        raise ValueError(f"No metrics.json files found under {run_root}")

    base_features = set(feature_names(extra_feature_base))
    runs = pd.DataFrame([_metrics_row(path, run_root) for path in metrics_paths])
    summary = _summary_frame(runs)
    selected = pd.DataFrame(
        row
        for path in metrics_paths
        for row in _selected_feature_rows(path, run_root, base_features)
    )
    selected_frequency = _selected_frequency_frame(selected)

    runs.to_csv(out_dir / "feature_ablation_runs.csv", index=False)
    summary.to_csv(out_dir / "feature_ablation_summary.csv", index=False)
    selected.to_csv(out_dir / "selected_features.csv", index=False)
    selected_frequency.to_csv(out_dir / "selected_feature_frequency.csv", index=False)

    top_extra = selected_frequency[selected_frequency["is_extra_feature"]].head(15)
    report = [
        "# Feature Ablation Summary",
        "",
        f"Run root: `{run_root}`",
        "",
        "## Mean Metrics",
        "",
        _markdown_table(
            summary,
            [
                "run",
                "feature_set",
                "classifier",
                "select_k",
                "n_runs",
                "accuracy_mean",
                "roc_auc_mean",
                "brier_score_mean",
                "expected_calibration_error_mean",
            ],
        ),
        "",
        "## Top Selected Extra Features",
        "",
        _markdown_table(top_extra, ["feature", "count", "runs"]),
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(_markdown_table(summary, ["run", "n_runs", "accuracy_mean", "roc_auc_mean"]))
    print(f"Wrote feature ablation summary to {out_dir.resolve()}")


def main() -> None:
    args = parse_args()
    summarize(Path(args.run_root), Path(args.out_dir), args.extra_feature_base)


if __name__ == "__main__":
    main()
