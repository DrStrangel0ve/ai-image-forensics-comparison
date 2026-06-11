from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from forensic_compare.utils import ensure_dir, read_json


METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize robustness evaluations across transformed dataset variants."
    )
    parser.add_argument(
        "--baseline",
        action="append",
        required=True,
        help="Baseline metrics in METHOD=PATH form. Repeat for each method.",
    )
    parser.add_argument(
        "--metrics",
        action="append",
        required=True,
        help="Variant metrics in VARIANT:METHOD=PATH form. Repeat for each variant/method.",
    )
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def _parse_baseline(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Baseline arguments must be METHOD=PATH, got {value!r}")
    method, path = value.split("=", 1)
    if not method or not path:
        raise ValueError(f"Baseline arguments must be METHOD=PATH, got {value!r}")
    return method, Path(path)


def _parse_variant_metric(value: str) -> tuple[str, str, Path]:
    if "=" not in value or ":" not in value.split("=", 1)[0]:
        raise ValueError(f"Metric arguments must be VARIANT:METHOD=PATH, got {value!r}")
    left, path = value.split("=", 1)
    variant, method = left.split(":", 1)
    if not variant or not method or not path:
        raise ValueError(f"Metric arguments must be VARIANT:METHOD=PATH, got {value!r}")
    return variant, method, Path(path)


def _metric_row(variant: str, method: str, path: Path, baseline_metrics: dict) -> dict:
    metrics = read_json(path)
    baseline = baseline_metrics[method]
    row = {
        "variant": variant,
        "method": method,
        "metrics_path": str(path),
        "n_samples": int(metrics.get("n_samples", metrics.get("n_target", 0))),
    }
    for column in METRIC_COLUMNS:
        value = float(metrics[column])
        row[column] = value
        row[f"{column}_delta"] = value - float(baseline[column])
    return row


def _format_cell(value) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _markdown_table(rows: list[dict], columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    baseline_paths = dict(_parse_baseline(value) for value in args.baseline)
    baseline_metrics = {method: read_json(path) for method, path in baseline_paths.items()}

    rows = [
        _metric_row(variant, method, path, baseline_metrics)
        for variant, method, path in map(_parse_variant_metric, args.metrics)
    ]
    rows = sorted(rows, key=lambda row: (row["variant"], row["method"]))

    columns = ["variant", "method", "accuracy", "accuracy_delta", "roc_auc", "roc_auc_delta"]
    with (out_dir / "robustness_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    report = [
        "# Robustness Summary",
        "",
        "Metric deltas are relative to each method's clean source-balanced validation score.",
        "",
        _markdown_table(rows, columns),
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(_markdown_table(rows, columns))
    print(f"Wrote robustness summary to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
