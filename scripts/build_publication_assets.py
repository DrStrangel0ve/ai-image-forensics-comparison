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
DINO_FUSION_ORDER = [
    "scp_fusion_v0",
    "source_calibrated",
    "scp_fusion_dinov2",
    "dinov2_source_calibrated",
]
DINO_FUSION_LABELS = {
    "scp_fusion_v0": "SCP-Fusion v0",
    "source_calibrated": "Source-calibrated",
    "scp_fusion_dinov2": "+ DINOv2",
    "dinov2_source_calibrated": "+ DINOv2 calibrated",
}
DINO_FUSION_COLORS = {
    "scp_fusion_v0": "#E45756",
    "source_calibrated": "#B279A2",
    "scp_fusion_dinov2": "#72B7B2",
    "dinov2_source_calibrated": "#4C78A8",
}
CLIP_FRONTIER_ORDER = [
    "scp_fusion_v0",
    "scp_fusion_dinov2",
    "scp_fusion_clip",
    "scp_fusion_all_foundation",
    "clip_standalone",
]
CLIP_FRONTIER_LABELS = {
    "scp_fusion_v0": "SCP-Fusion v0",
    "scp_fusion_dinov2": "+ DINOv2",
    "scp_fusion_clip": "+ CLIP",
    "scp_fusion_all_foundation": "+ CLIP + DINOv2",
    "clip_standalone": "CLIP alone",
}
CLIP_FRONTIER_COLORS = {
    "scp_fusion_v0": "#E45756",
    "scp_fusion_dinov2": "#72B7B2",
    "scp_fusion_clip": "#54A24B",
    "scp_fusion_all_foundation": "#B279A2",
    "clip_standalone": "#4C78A8",
}
REVERSE_FUSION_ORDER = [
    "score_fusion_all6",
    "score_fusion_all6_temp_balanced",
    "score_fusion_all6_c01",
    "score_fusion_all6_c01_temp_balanced",
    "score_fusion_all6_c003",
    "score_fusion_all6_dropout_mean_r35x8",
    "score_fusion_all6_dropout_mean_r35x8_temp_balanced",
]
REVERSE_FUSION_LABELS = {
    "score_fusion_all6": "baseline",
    "score_fusion_all6_temp_balanced": "baseline + temp",
    "score_fusion_all6_c01": "C=.1",
    "score_fusion_all6_c01_temp_balanced": "C=.1 + temp",
    "score_fusion_all6_c003": "C=.03",
    "score_fusion_all6_dropout_mean_r35x8": "dropout",
    "score_fusion_all6_dropout_mean_r35x8_temp_balanced": "dropout + temp",
}
REVERSE_FUSION_COLORS = {
    "score_fusion_all6": "#9E9E9E",
    "score_fusion_all6_temp_balanced": "#B279A2",
    "score_fusion_all6_c01": "#F58518",
    "score_fusion_all6_c01_temp_balanced": "#FFBF79",
    "score_fusion_all6_c003": "#54A24B",
    "score_fusion_all6_dropout_mean_r35x8": "#4C78A8",
    "score_fusion_all6_dropout_mean_r35x8_temp_balanced": "#72B7B2",
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
    parser.add_argument(
        "--score-fusion-dinov2-calibration",
        default="reports/assets/score_fusion_dinov2_calibration_summary.csv",
    )
    parser.add_argument(
        "--score-fusion-dinov2-triage-5pct",
        default="reports/assets/score_fusion_dinov2_source_holdout_triage_5pct.csv",
    )
    parser.add_argument(
        "--score-fusion-dinov2-triage-10pct",
        default="reports/assets/score_fusion_dinov2_source_holdout_triage_10pct.csv",
    )
    parser.add_argument(
        "--score-fusion-clip-calibration",
        default="reports/assets/score_fusion_clip_calibration_summary.csv",
    )
    parser.add_argument(
        "--score-fusion-clip-triage-5pct",
        default="reports/assets/score_fusion_clip_source_holdout_triage_5pct.csv",
    )
    parser.add_argument(
        "--score-fusion-clip-triage-10pct",
        default="reports/assets/score_fusion_clip_source_holdout_triage_10pct.csv",
    )
    parser.add_argument(
        "--reverse-fusion-regularization",
        default="reports/assets/ms_cocoai_to_ishu_reverse_fusion_regularization_mean_metrics.csv",
    )
    parser.add_argument(
        "--reverse-fusion-thresholds",
        default="reports/assets/ms_cocoai_to_ishu_reverse_fusion_regularization_threshold_means.csv",
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


def _dino_fusion_ordered(frame: pd.DataFrame) -> pd.DataFrame:
    ordered_methods = [method for method in DINO_FUSION_ORDER if method in set(frame["method"])]
    return frame.set_index("method").loc[ordered_methods].reset_index()


def _dino_fusion_labels(frame: pd.DataFrame) -> list[str]:
    return [DINO_FUSION_LABELS.get(method, method) for method in frame["method"]]


def _dino_fusion_colors(frame: pd.DataFrame) -> list[str]:
    return [DINO_FUSION_COLORS.get(method, "#777777") for method in frame["method"]]


def _dino_triage_frame(path: Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["score_mode"] == "raw"].copy()
    frame["operating_point"] = label
    return _dino_fusion_ordered(frame)


def build_score_fusion_dinov2_gain(
    calibration_summary: Path,
    triage_5pct: Path,
    triage_10pct: Path,
    out_dir: Path,
    dpi: int,
) -> Path:
    calibration = _dino_fusion_ordered(pd.read_csv(calibration_summary))
    triage = pd.concat(
        [
            _dino_triage_frame(triage_5pct, "5% budget"),
            _dino_triage_frame(triage_10pct, "10% budget"),
        ],
        ignore_index=True,
    )
    labels = _dino_fusion_labels(calibration)
    colors = _dino_fusion_colors(calibration)
    x = np.arange(len(calibration))

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4), dpi=dpi)
    for ax, column, title, ylabel, ylim in [
        (axes[0, 0], "mean_roc_auc", "Transfer Ranking", "AUC", (0.68, 0.77)),
        (axes[0, 1], "mean_brier_score", "Probability Error", "Brier score", (0.28, 0.34)),
    ]:
        values = calibration[column].astype(float)
        ax.bar(x, values, color=colors, edgecolor="#222222", linewidth=0.4)
        ax.set_title(title, fontsize=10, pad=8)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.25)
        _annotate_bars(ax, values, offset=(ylim[1] - ylim[0]) * 0.02)

    for ax, operating_point in [(axes[1, 0], "5% budget"), (axes[1, 1], "10% budget")]:
        frame = triage[triage["operating_point"] == operating_point].copy()
        frame = _dino_fusion_ordered(frame)
        x = np.arange(len(frame))
        width = 0.34
        ax.bar(
            x - width / 2,
            frame["mean_test_coverage"],
            width,
            label="coverage",
            color="#72B7B2",
            edgecolor="#222222",
            linewidth=0.4,
        )
        ax.bar(
            x + width / 2,
            frame["mean_test_triage_accuracy"],
            width,
            label="triage accuracy",
            color="#4C78A8",
            edgecolor="#222222",
            linewidth=0.4,
        )
        ax.set_title(f"Source-Heldout Triage ({operating_point})", fontsize=10, pad=8)
        ax.set_ylabel("fraction")
        ax.set_xticks(x)
        ax.set_xticklabels(_dino_fusion_labels(frame), rotation=25, ha="right")
        ax.set_ylim(0.0, 0.9)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("DINOv2 strengthens SCP-Fusion under generator shift", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    out_path = out_dir / "publication_score_fusion_dinov2_gain.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _clip_frontier_ordered(frame: pd.DataFrame) -> pd.DataFrame:
    ordered_methods = [method for method in CLIP_FRONTIER_ORDER if method in set(frame["method"])]
    return frame.set_index("method").loc[ordered_methods].reset_index()


def _clip_frontier_labels(frame: pd.DataFrame) -> list[str]:
    return [CLIP_FRONTIER_LABELS.get(method, method) for method in frame["method"]]


def _clip_frontier_colors(frame: pd.DataFrame) -> list[str]:
    return [CLIP_FRONTIER_COLORS.get(method, "#777777") for method in frame["method"]]


def _clip_triage_frame(path: Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame = frame[frame["score_mode"] == "raw"].copy()
    frame["operating_point"] = label
    return _clip_frontier_ordered(frame)


def build_score_fusion_clip_frontier(
    calibration_summary: Path,
    triage_5pct: Path,
    triage_10pct: Path,
    out_dir: Path,
    dpi: int,
) -> Path:
    calibration = _clip_frontier_ordered(pd.read_csv(calibration_summary))
    triage = pd.concat(
        [
            _clip_triage_frame(triage_5pct, "5% budget"),
            _clip_triage_frame(triage_10pct, "10% budget"),
        ],
        ignore_index=True,
    )
    labels = _clip_frontier_labels(calibration)
    colors = _clip_frontier_colors(calibration)
    x = np.arange(len(calibration))

    fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.4), dpi=dpi)
    for ax, column, title, ylabel, ylim in [
        (axes[0, 0], "mean_roc_auc", "Transfer Ranking", "AUC", (0.68, 0.90)),
        (axes[0, 1], "mean_accuracy", "Default Decision Accuracy", "accuracy", (0.54, 0.66)),
    ]:
        values = calibration[column].astype(float)
        ax.bar(x, values, color=colors, edgecolor="#222222", linewidth=0.4)
        ax.set_title(title, fontsize=10, pad=8)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right")
        ax.set_ylim(*ylim)
        ax.grid(axis="y", alpha=0.25)
        _annotate_bars(ax, values, offset=(ylim[1] - ylim[0]) * 0.025)

    for ax, operating_point in [(axes[1, 0], "5% budget"), (axes[1, 1], "10% budget")]:
        frame = _clip_frontier_ordered(triage[triage["operating_point"] == operating_point].copy())
        x = np.arange(len(frame))
        width = 0.34
        ax.bar(
            x - width / 2,
            frame["mean_test_coverage"],
            width,
            label="coverage",
            color="#72B7B2",
            edgecolor="#222222",
            linewidth=0.4,
        )
        ax.bar(
            x + width / 2,
            frame["mean_test_triage_accuracy"],
            width,
            label="triage accuracy",
            color="#4C78A8",
            edgecolor="#222222",
            linewidth=0.4,
        )
        ax.set_title(f"Source-Heldout Triage ({operating_point})", fontsize=10, pad=8)
        ax.set_ylabel("fraction")
        ax.set_xticks(x)
        ax.set_xticklabels(_clip_frontier_labels(frame), rotation=25, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)

    fig.suptitle("CLIP sets the current transfer and triage frontier", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    out_path = out_dir / "publication_score_fusion_clip_frontier.png"
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def _reverse_fusion_ordered(frame: pd.DataFrame) -> pd.DataFrame:
    ordered_configs = [config for config in REVERSE_FUSION_ORDER if config in set(frame["config"])]
    return frame.set_index("config").loc[ordered_configs].reset_index()


def _reverse_fusion_labels(frame: pd.DataFrame) -> list[str]:
    return [REVERSE_FUSION_LABELS.get(config, config) for config in frame["config"]]


def build_reverse_fusion_tradeoff(
    mean_metrics_path: Path,
    threshold_means_path: Path,
    out_dir: Path,
    dpi: int,
) -> Path:
    metrics = pd.read_csv(mean_metrics_path)
    metrics = metrics[metrics["split"] == "ms_cocoai_to_ishu_test"].copy()
    metrics = _reverse_fusion_ordered(metrics)
    thresholds = _reverse_fusion_ordered(pd.read_csv(threshold_means_path))
    labels = _reverse_fusion_labels(metrics)
    colors = [REVERSE_FUSION_COLORS.get(config, "#777777") for config in metrics["config"]]

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), dpi=dpi)
    ax = axes[0]
    sizes = 1200.0 * metrics["accuracy"].astype(float).to_numpy()
    ax.scatter(
        metrics["brier"],
        metrics["auc"],
        s=sizes,
        c=colors,
        edgecolor="#222222",
        linewidth=0.5,
        alpha=0.92,
    )
    annotation_offsets = {
        "score_fusion_all6_dropout_mean_r35x8": (10, 8),
        "score_fusion_all6_dropout_mean_r35x8_temp_balanced": (-62, -10),
        "score_fusion_all6_temp_balanced": (8, 4),
        "score_fusion_all6": (8, 4),
    }
    for _, row in metrics.iterrows():
        offset = annotation_offsets.get(row["config"], (5, 4))
        ax.annotate(
            REVERSE_FUSION_LABELS.get(row["config"], row["config"]),
            (float(row["brier"]), float(row["auc"])),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_title("Reverse transfer fusion tradeoff", fontsize=10, pad=8)
    ax.set_xlabel("Brier score (lower is better)")
    ax.set_ylabel("AUC (higher is better)")
    ax.set_xlim(float(metrics["brier"].min()) - 0.004, float(metrics["brier"].max()) + 0.008)
    ax.set_ylim(float(metrics["auc"].min()) - 0.001, float(metrics["auc"].max()) + 0.001)
    ax.grid(alpha=0.25)

    ax = axes[1]
    x = np.arange(len(thresholds))
    width = 0.35
    ax.bar(
        x - width / 2,
        thresholds["default_accuracy"].astype(float),
        width,
        label="default",
        color="#9E9E9E",
        edgecolor="#222222",
        linewidth=0.4,
    )
    ax.bar(
        x + width / 2,
        thresholds["clean_threshold_accuracy"].astype(float),
        width,
        label="source threshold",
        color="#4C78A8",
        edgecolor="#222222",
        linewidth=0.4,
    )
    ax.set_title("Thresholded decision quality", fontsize=10, pad=8)
    ax.set_ylabel("accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(_reverse_fusion_labels(thresholds), rotation=28, ha="right")
    ax.set_ylim(0.60, 0.70)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)

    fig.suptitle("MS COCOAI to Ishu: fusion ranking vs calibration", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    out_path = out_dir / "publication_reverse_fusion_tradeoff.png"
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
        build_score_fusion_dinov2_gain(
            Path(args.score_fusion_dinov2_calibration),
            Path(args.score_fusion_dinov2_triage_5pct),
            Path(args.score_fusion_dinov2_triage_10pct),
            out_dir,
            args.dpi,
        ),
        build_score_fusion_clip_frontier(
            Path(args.score_fusion_clip_calibration),
            Path(args.score_fusion_clip_triage_5pct),
            Path(args.score_fusion_clip_triage_10pct),
            out_dir,
            args.dpi,
        ),
        build_reverse_fusion_tradeoff(
            Path(args.reverse_fusion_regularization),
            Path(args.reverse_fusion_thresholds),
            out_dir,
            args.dpi,
        ),
    ]
    for output in outputs:
        print(output.resolve())


if __name__ == "__main__":
    main()
