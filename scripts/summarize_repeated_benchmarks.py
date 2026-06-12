from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from forensic_compare.metrics import bootstrap_mean_ci
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
    parser.add_argument("--ci-confidence", type=float, default=0.95)
    parser.add_argument("--ci-resamples", type=int, default=2000)
    parser.add_argument("--ci-seed", type=int, default=0)
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


def _summary_frame(
    all_runs: pd.DataFrame,
    confidence: float,
    n_resamples: int,
    seed: int,
) -> pd.DataFrame:
    grouped = all_runs.groupby("method", sort=True)
    rows = []
    for method, group in grouped:
        row = {"method": method, "n_runs": int(len(group))}
        for index, metric in enumerate(METRIC_COLUMNS):
            interval = bootstrap_mean_ci(
                group[metric].to_numpy(dtype=float),
                confidence=confidence,
                n_resamples=n_resamples,
                seed=seed + index,
            )
            row[f"{metric}_mean"] = interval["mean"]
            row[f"{metric}_ci_low"] = interval["ci_low"]
            row[f"{metric}_ci_high"] = interval["ci_high"]
            row[f"{metric}_std"] = float(group[metric].std(ddof=0))
            row[f"{metric}_min"] = float(group[metric].min())
            row[f"{metric}_max"] = float(group[metric].max())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(
    comparisons: list[tuple[str, Path]],
    out_dir: Path,
    ci_confidence: float = 0.95,
    ci_resamples: int = 2000,
    ci_seed: int = 0,
) -> None:
    out_dir = ensure_dir(out_dir)
    frames = []
    for run_name, path in comparisons:
        frame = pd.read_csv(path)
        frame.insert(0, "run", run_name)
        frames.append(frame)
    all_runs = pd.concat(frames, ignore_index=True)
    all_runs.to_csv(out_dir / "repeated_runs.csv", index=False)

    summary = _summary_frame(all_runs, ci_confidence, ci_resamples, ci_seed)
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
        "accuracy_ci_low",
        "accuracy_ci_high",
        "accuracy_std",
        "accuracy_min",
        "accuracy_max",
        "roc_auc_mean",
        "roc_auc_ci_low",
        "roc_auc_ci_high",
        "roc_auc_std",
        "roc_auc_min",
        "roc_auc_max",
    ]
    report = [
        "# Repeated Benchmark Summary",
        "",
        "Metric standard deviations use population standard deviation across the supplied runs.",
        f"Confidence intervals are `{ci_confidence:.0%}` deterministic bootstrap intervals over supplied runs with `{ci_resamples}` resamples.",
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


def main() -> None:
    args = parse_args()
    summarize(
        list(map(_parse_comparison, args.comparison)),
        Path(args.out_dir),
        ci_confidence=args.ci_confidence,
        ci_resamples=args.ci_resamples,
        ci_seed=args.ci_seed,
    )


if __name__ == "__main__":
    main()
