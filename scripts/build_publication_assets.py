from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forensic_compare.utils import ensure_dir


METHOD_ORDER = [
    "combined_v3",
    "resnet18",
    "physics_guided",
    "convnext_tiny_frozen",
    "scp_fusion_v0",
]
METHOD_LABELS = {
    "combined_v3": "combined_v3",
    "resnet18": "ResNet-18",
    "physics_guided": "Physics-guided",
    "convnext_tiny_frozen": "Frozen ConvNeXt",
    "scp_fusion_v0": "SCP-Fusion v0",
}
METHOD_COLORS = {
    "combined_v3": "#4C78A8",
    "resnet18": "#F58518",
    "physics_guided": "#54A24B",
    "convnext_tiny_frozen": "#B279A2",
    "scp_fusion_v0": "#E45756",
}
SCORE_FUSION_ORDER = ["scp_fusion_v0", "branch_dropout", "source_calibrated"]
SCORE_FUSION_LABELS = {
    "scp_fusion_v0": "SCP-Fusion v0",
    "branch_dropout": "Branch dropout",
    "source_calibrated": "Source-calibrated",
}
SCORE_FUSION_COLORS = {
    "scp_fusion_v0": "#E45756",
    "branch_dropout": "#72B7B2",
    "source_calibrated": "#B279A2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build paper/poster figures from checked-in benchmark summary CSVs."
    )
    parser.add_argument(
        "--calibration-summary",
        default="reports/assets/calibration_summary_ishu_ms_cocoai_all4.csv",
    )
    parser.add_argument(
        "--source-heldout-calibration",
        default="reports/assets/source_holdout_calibration_summary_ishu_ms_cocoai_all4.csv",
    )
    parser.add_argument(
        "--triage-5pct",
        default="reports/assets/source_holdout_triage_summary_ishu_ms_cocoai_all4_5pct.csv",
    )
    parser.add_argument(
        "--triage-10pct",
        default="reports/assets/source_holdout_triage_summary_ishu_ms_cocoai_all4_10pct.csv",
    )
    parser.add_argument(
        "--score-fusion-tuned-triage",
        default="reports/assets/score_fusion_source_holdout_triage_tuned_summary.csv",
    )
    parser.add_argument("--out-dir", default="reports/assets")
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def _ordered(frame: pd.DataFrame) -> pd.DataFrame:
    ordered_methods = [method for method in METHOD_ORDER if method in set(frame["method"])]
    return frame.set_index("method").loc[ordered_methods].reset_index()


def _labels(frame: pd.DataFrame) -> list[str]:
    return [METHOD_LABELS.get(method, method) for method in frame["method"]]


def _colors(frame: pd.DataFrame) -> list[str]:
    return [METHOD_COLORS.get(method, "#777777") for method in frame["method"]]


def _annotate_bars(ax: plt.Axes, values: pd.Series, offset: float = 0.01) -> None:
    for index, value in enumerate(values):
        if pd.isna(value):
            continue
        ax.text(index, float(value) + offset, f"{float(value):.2f}", ha="center", va="bottom", fontsize=8)


def _bar_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    column: str,
    title: str,
    ylabel: str,
    ylim: tuple[float, float] | None = None,
) -> None:
    values = frame[column].astype(float)
    ax.bar(np.arange(len(frame)), values, color=_colors(frame), edgecolor="#222222", linewidth=0.4)
    ax.set_title(title, fontsize=10, pad=8)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(len(frame)))
    ax.set_xticklabels(_labels(frame), rotation=30, ha="right")
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.25)
    _annotate_bars(ax, values)


def build_cross_domain_calibration(calibration_summary: Path, out_dir: Path, dpi: int) -> Path:
    frame = _ordered(pd.read_csv(calibration_summary))
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), dpi=dpi)
    _bar_panel(axes[0, 0], frame, "mean_accuracy", "Default Accuracy", "accuracy", (0.0, 0.8))
    _bar_panel(axes[0, 1], frame, "mean_roc_auc", "Ranking AUC", "AUC", (0.0, 0.9))
    _bar_panel(axes[1, 0], frame, "mean_brier_score", "Brier Score", "lower is better", (0.0, 0.45))
    _bar_panel(
        axes[1, 1],
        frame,
        "mean_expected_calibration_error",
        "Expected Calibration Error",
        "lower is better",
        (0.0, 0.4),
    )
    fig.suptitle("Ishu to MS COCOAI transfer: ranking and calibration", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    out_path = out_dir / "publication_cross_domain_calibration.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def build_source_heldout_calibration(source_calibration: Path, out_dir: Path, dpi: int) -> Path:
    frame = pd.read_csv(source_calibration)
    frame = frame[frame["calibrator"] == "temperature_balanced"].copy()
    frame = _ordered(frame)
    x = np.arange(len(frame))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), dpi=dpi)
    for ax, raw_col, cal_col, title in [
        (
            axes[0],
            "mean_raw_brier_score",
            "mean_calibrated_brier_score",
            "Brier score",
        ),
        (
            axes[1],
            "mean_raw_ece",
            "mean_calibrated_ece",
            "Expected calibration error",
        ),
    ]:
        ax.bar(x - width / 2, frame[raw_col], width, label="raw", color="#9E9E9E")
        ax.bar(x + width / 2, frame[cal_col], width, label="balanced temperature", color="#4C78A8")
        ax.set_title(title, fontsize=10, pad=8)
        ax.set_ylabel("lower is better")
        ax.set_xticks(x)
        ax.set_xticklabels(_labels(frame), rotation=30, ha="right")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Source-heldout post-hoc calibration", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    out_path = out_dir / "publication_source_heldout_calibration.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _triage_frame(path: Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["score_mode"] == "raw"].copy()
    frame["operating_point"] = label
    return frame


def build_triage_operating_points(triage_5pct: Path, triage_10pct: Path, out_dir: Path, dpi: int) -> Path:
    frame = pd.concat(
        [
            _triage_frame(triage_5pct, "5% calibration budget"),
            _triage_frame(triage_10pct, "10% calibration budget"),
        ],
        ignore_index=True,
    )
    frame = frame[frame["method"].isin(METHOD_ORDER)].copy()
    frame["method"] = pd.Categorical(frame["method"], categories=METHOD_ORDER, ordered=True)
    frame = frame.sort_values(["method", "operating_point"])

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), dpi=dpi)
    for ax, column, title, ylabel in [
        (axes[0], "mean_test_coverage", "Decided Coverage", "fraction of target images"),
        (axes[1], "mean_test_triage_accuracy", "Accuracy on Decided Images", "accuracy"),
    ]:
        pivot = frame.pivot(index="method", columns="operating_point", values=column).loc[METHOD_ORDER]
        x = np.arange(len(pivot))
        width = 0.34
        ax.bar(x - width / 2, pivot["5% calibration budget"], width, label="5%", color="#4C78A8")
        ax.bar(x + width / 2, pivot["10% calibration budget"], width, label="10%", color="#F58518")
        ax.set_title(title, fontsize=10, pad=8)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS.get(method, method) for method in pivot.index], rotation=30, ha="right")
        ax.set_ylim(0.0, 0.85)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(title="error budget", fontsize=8, title_fontsize=8)
    fig.suptitle("Source-heldout triage operating points", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    out_path = out_dir / "publication_triage_operating_points.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _score_fusion_ordered(frame: pd.DataFrame) -> pd.DataFrame:
    ordered_methods = [method for method in SCORE_FUSION_ORDER if method in set(frame["method"])]
    return frame.set_index("method").loc[ordered_methods].reset_index()


def build_score_fusion_tuned_triage(tuned_summary: Path, out_dir: Path, dpi: int) -> Path:
    frame = _score_fusion_ordered(pd.read_csv(tuned_summary))
    labels = [
        f"{SCORE_FUSION_LABELS.get(row.method, row.method)}\n{row.selected_score_modes}"
        for row in frame.itertuples(index=False)
    ]
    colors = [SCORE_FUSION_COLORS.get(method, "#777777") for method in frame["method"]]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), dpi=dpi)

    utility = frame["mean_test_utility"].astype(float)
    lower = utility - frame["mean_test_utility_ci_low"].astype(float)
    upper = frame["mean_test_utility_ci_high"].astype(float) - utility
    axes[0].bar(np.arange(len(frame)), utility, color=colors, edgecolor="#222222", linewidth=0.4)
    axes[0].errorbar(
        np.arange(len(frame)),
        utility,
        yerr=[lower, upper],
        fmt="none",
        ecolor="#222222",
        elinewidth=0.8,
        capsize=3,
    )
    axes[0].axhline(0.0, color="#333333", linewidth=0.8)
    axes[0].set_title("Selected Triage Utility", fontsize=10, pad=8)
    axes[0].set_ylabel("source-heldout utility")
    axes[0].set_xticks(np.arange(len(frame)))
    axes[0].set_xticklabels(labels, rotation=25, ha="right")
    axes[0].grid(axis="y", alpha=0.25)
    _annotate_bars(axes[0], utility, offset=0.006)

    component_columns = [
        ("mean_test_fake_detection", "fake\ndetect"),
        ("mean_test_real_clearance", "real\nclear"),
        ("mean_test_real_fpr", "real\nFPR"),
        ("mean_test_fake_false_clearance", "fake false\nclear"),
    ]
    x = np.arange(len(component_columns))
    width = 0.22
    offsets = np.linspace(-width, width, len(frame))
    for offset, row, color in zip(offsets, frame.itertuples(index=False), colors, strict=True):
        values = [float(getattr(row, column)) for column, _label in component_columns]
        axes[1].bar(
            x + offset,
            values,
            width,
            label=SCORE_FUSION_LABELS.get(row.method, row.method),
            color=color,
            edgecolor="#222222",
            linewidth=0.4,
        )
    axes[1].set_title("Utility Components", fontsize=10, pad=8)
    axes[1].set_ylabel("fraction of target images")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([label for _column, label in component_columns])
    axes[1].set_ylim(0.0, 0.55)
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(fontsize=8)

    fig.suptitle("Source-heldout score-fusion triage tuning", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    out_path = out_dir / "publication_score_fusion_tuned_triage.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    outputs = [
        build_cross_domain_calibration(Path(args.calibration_summary), out_dir, args.dpi),
        build_source_heldout_calibration(Path(args.source_heldout_calibration), out_dir, args.dpi),
        build_triage_operating_points(Path(args.triage_5pct), Path(args.triage_10pct), out_dir, args.dpi),
        build_score_fusion_tuned_triage(Path(args.score_fusion_tuned_triage), out_dir, args.dpi),
    ]
    for output in outputs:
        print(output.resolve())


if __name__ == "__main__":
    main()
