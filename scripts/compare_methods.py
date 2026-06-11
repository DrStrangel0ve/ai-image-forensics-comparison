from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from forensic_compare.utils import ensure_dir, read_json


METRIC_COLUMNS = ["accuracy", "precision", "recall", "f1", "roc_auc", "n_samples"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare neural and photometric method metrics.")
    parser.add_argument("--neural", default="runs/resnet18/metrics.json")
    parser.add_argument("--photometric", default="runs/photometric/metrics.json")
    parser.add_argument("--out-dir", default="runs/comparison")
    return parser.parse_args()


def _row(name: str, metrics_path: str | Path) -> dict:
    metrics = read_json(metrics_path)
    row = {"method": name, "metrics_path": str(metrics_path)}
    for column in METRIC_COLUMNS:
        row[column] = metrics.get(column)
    return row


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    rows = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append("" if pd.isna(value) else str(value))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    rows = [
        _row("neural_net", args.neural),
        _row("photometric_normal_consistency", args.photometric),
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(out_dir / "comparison.csv", index=False)

    best = frame.sort_values("accuracy", ascending=False).iloc[0]
    report = [
        "# Method Comparison",
        "",
        _markdown_table(frame),
        "",
        f"Best accuracy: **{best['method']}** at **{best['accuracy']:.4f}**.",
        "",
        "The photometric baseline is a single-image normal-consistency proxy. "
        "It is useful as a physics-inspired check, but it is not a substitute "
        "for calibrated multi-light photometric stereo data.",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(frame.to_string(index=False))
    print(f"Wrote comparison report to {(out_dir / 'report.md').resolve()}")


if __name__ == "__main__":
    main()
