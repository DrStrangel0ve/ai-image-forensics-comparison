from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_dfrws_poster_figure_builder_writes_panels_and_source_tables(tmp_path: Path) -> None:
    core_results = tmp_path / "publication_core_results.csv"
    out_dir = tmp_path / "assets"
    report_out = tmp_path / "dfrws_poster_native_figures.md"

    rows = [
        ("ishu_same_combined_v3", "combined_v3", 0.8246, 0.8942, None, None),
        ("ishu_same_resnet18", "ResNet-18", 0.8246, 0.8927, None, None),
        ("ishu_same_physics_guided", "Physics-guided", 0.8450, 0.9177, None, None),
        ("ishu_to_ms_physics_guided", "Physics-guided", 0.6060, 0.6637, 0.2707, None),
        ("ishu_to_ms_convnext_tiny_frozen", "Frozen ConvNeXt", 0.6163, 0.7139, 0.1870, None),
        ("ishu_to_ms_scp_fusion_all_foundation", "SCP-Fusion all-foundation", 0.6163, 0.7995, 0.1323, None),
        ("ishu_to_ms_clip_standalone", "Frozen CLIP", 0.6363, 0.8641, 0.1630, None),
        ("ishu_to_ms_triage5_clip_standalone", "Frozen CLIP triage", None, None, None, 0.4747),
        ("ms_to_ishu_tuned_fusion_constraint_sweep_best", "clean", 0.7632, 0.8361, 0.5175, None),
        ("ms_to_ishu_tuned_fusion_native_tiling_best", "tiled-v3", 0.7749, 0.8472, 0.5468, None),
        ("ms_to_ishu_tuned_fusion_jpeg70", "JPEG70", 0.7661, 0.8485, 0.4678, None),
        ("ms_to_ishu_tuned_fusion_jpeg50", "JPEG50", 0.7515, 0.8309, 0.4240, None),
        ("ms_to_ishu_tuned_fusion_jpeg30", "JPEG30", 0.7076, 0.8167, 0.3450, None),
        ("ms_to_ishu_tuned_fusion_noise3", "noise3", 0.7690, 0.8704, 0.4708, None),
        ("ms_to_ishu_tuned_fusion_social_square", "social", 0.7778, 0.8474, 0.5088, None),
        ("ms_to_ishu_tuned_fusion_social_720p", "social720", 0.7602, 0.8506, 0.4678, None),
        ("ms_to_ishu_tuned_fusion_crop85", "crop85", 0.7251, 0.8227, 0.5205, None),
        ("ms_to_ishu_tuned_fusion_screenshot", "screen", 0.7310, 0.7965, 0.5263, None),
        ("ms_to_ishu_tuned_fusion_blur1", "blur1", 0.7105, 0.7872, 0.5585, None),
        ("ms_to_ishu_tuned_fusion_resize_half", "resize", 0.7164, 0.7816, 0.5526, None),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "finding_id",
            "method",
            "accuracy",
            "auc",
            "predicted_fake_rate",
            "coverage",
        ],
    )
    frame["setting"] = "setting"
    frame["brier"] = 0.25
    frame["ece"] = 0.20
    frame["decided_accuracy"] = 0.9261
    frame["source"] = "source"
    frame["interpretation"] = "interpretation"
    frame.to_csv(core_results, index=False)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_dfrws_poster_figures.py"),
            "--core-results",
            str(core_results),
            "--out-dir",
            str(out_dir),
            "--report-out",
            str(report_out),
            "--dpi",
            "90",
        ],
        cwd=ROOT,
        check=True,
    )

    transfer_csv = pd.read_csv(out_dir / "dfrws_poster_transfer_panel.csv")
    robustness_csv = pd.read_csv(out_dir / "dfrws_poster_robustness_panel.csv")
    report = report_out.read_text(encoding="utf-8")
    transfer_svg = (out_dir / "dfrws_poster_transfer_panel.svg").read_text(encoding="utf-8")

    assert (out_dir / "dfrws_poster_transfer_panel.png").stat().st_size > 0
    assert (out_dir / "dfrws_poster_robustness_panel.png").stat().st_size > 0
    assert len(transfer_csv) == 8
    assert len(robustness_csv) == 12
    assert "Frozen CLIP" in set(transfer_csv["label"])
    assert "tiled-v3" in set(robustness_csv["label"])
    assert "JPEG30" in set(robustness_csv["label"])
    assert "social720" in set(robustness_csv["label"])
    assert "DFRWS Poster-Native Figure Pack" in report
    assert "Frozen CLIP" in transfer_svg
