from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from forensic_compare.utils import ensure_dir


METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize repeated benchmark comparison CSVs.")
    parser.add_argument(
        "--comparison",
        action="append",
        required=True,
        help="Comparison CSV in RUN=PATH form. Repeat for each run.",
    )
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def _parse_comparison(value: str) -> tuple[str, Path]:
    if "=" not in value:
        path = Path(value)
        return path.parent.parent.name or path.stem, path
    run_name, path = value.split("=", 1)
    if not run_name or not path:
        raise ValueError(f"Comparison arguments must be RUN=PATH, got {value!r}")
    return run_name, Path(path)


def _format_float(value: float) -> str:
    return f"{value:.4f}"


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(_format_float(value))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    frames = []
    for run_name, path in map(_parse_comparison, args.comparison):
        frame = pd.read_csv(path)
        frame.insert(0, "run", run_name)
        frames.append(frame)
    all_runs = pd.concat(frames, ignore_index=True)
    all_runs.to_csv(out_dir / "repeated_runs.csv", index=False)

    grouped = all_runs.groupby("method", sort=True)
    rows = []
    for method, group in grouped:
        row = {"method": method, "n_runs": int(len(group))}
        for metric in METRIC_COLUMNS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=0))
            row[f"{metric}_min"] = float(group[metric].min())
            row[f"{metric}_max"] = float(group[metric].max())
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(out_dir / "repeated_summary.csv", index=False)

    win_rows = []
    for metric in ["accuracy", "roc_auc"]:
        counts = all_runs.loc[all_runs.groupby("run")[metric].idxmax(), "method"].value_counts()
        for method in sorted(all_runs["method"].unique()):
            win_rows.append({"metric": metric, "method": method, "wins": int(counts.get(method, 0))})
    wins = pd.DataFrame(win_rows)
    wins.to_csv(out_dir / "wins.csv", index=False)

    summary_columns = [
        "method",
        "n_runs",
        "accuracy_mean",
        "accuracy_std",
        "accuracy_min",
        "accuracy_max",
        "roc_auc_mean",
        "roc_auc_std",
        "roc_auc_min",
        "roc_auc_max",
    ]
    report = [
        "# Repeated Benchmark Summary",
        "",
        "Metric standard deviations use population standard deviation across the supplied runs.",
        "",
        _markdown_table(summary, summary_columns),
        "",
        "## Per-Run Winners",
        "",
        _markdown_table(wins, ["metric", "method", "wins"]),
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(_markdown_table(summary, summary_columns))
    print(f"Wrote repeated benchmark summary to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
