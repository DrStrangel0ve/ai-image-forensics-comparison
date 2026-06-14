from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def test_dfrws_poster_package_lint_passes_generated_fixture(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    scripts_dir = repo_root / "scripts"
    reports_dir = repo_root / "reports"
    assets_dir = reports_dir / "assets"
    scripts_dir.mkdir(parents=True)
    assets_dir.mkdir(parents=True)
    for script in [
        "build_dfrws_poster_brief.py",
        "build_dfrws_poster_figures.py",
        "build_source_stress_figure.py",
        "source_stress_utils.py",
        "tiled_dino_tradeoff_utils.py",
    ]:
        shutil.copyfile(ROOT / "scripts" / script, scripts_dir / script)

    rows = [
        ("ishu_same_combined_v3", "combined_v3", 0.8246, 0.8942, None, None),
        ("ishu_same_resnet18", "ResNet-18", 0.8246, 0.8927, None, None),
        ("ishu_same_physics_guided", "Physics-guided", 0.8450, 0.9177, None, None),
        ("ishu_to_ms_physics_guided", "Physics-guided", 0.6060, 0.6637, 0.2707, None),
        ("ishu_to_ms_convnext_tiny_frozen", "Frozen ConvNeXt", 0.6163, 0.7139, 0.1870, None),
        ("ishu_to_ms_scp_fusion_all_foundation", "SCP-Fusion all-foundation", 0.6163, 0.7995, 0.1323, None),
        ("ishu_to_ms_clip_standalone", "Frozen CLIP", 0.6363, 0.8641, 0.1630, None),
        ("ishu_to_ms_triage5_clip_standalone", "Frozen CLIP triage", None, None, None, 0.4747),
        ("ms_to_ishu_branch_dropout_auc", "dropout", 0.6520, 0.8406, 0.8158, None),
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
    core = pd.DataFrame(
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
    core["setting"] = "setting"
    core["brier"] = 0.25
    core["ece"] = 0.20
    core["decided_accuracy"] = 0.9261
    core["source"] = "source"
    core["interpretation"] = "interpretation"
    core_results = assets_dir / "publication_core_results.csv"
    core.to_csv(core_results, index=False)

    claim_matrix = assets_dir / "claim_evidence_matrix.csv"
    pd.DataFrame(
        {
            "claim_id": ["clip_transfer_frontier"],
            "claim": ["claim"],
            "submission_use": ["DFRWS poster lead"],
            "status": ["ready"],
            "evidence_finding_ids": ["ishu_to_ms_clip_standalone"],
            "evidence_summary": ["summary"],
            "primary_artifact": ["artifact"],
            "risk_or_caveat": ["caveat"],
            "next_action": ["action"],
        }
    ).to_csv(claim_matrix, index=False)
    tiled_dino_tradeoff = assets_dir / "tiled_dinov2_calibration_tradeoff.csv"
    pd.DataFrame(
        [
            {
                "variant": variant,
                "score_mode": score_mode,
                "target_accuracy_mean_delta_vs_global": acc_delta,
                "target_roc_auc_mean_delta_vs_global": auc_delta,
                "target_brier_score_mean_improved_vs_global": brier_improved,
                "target_expected_calibration_error_mean_improved_vs_global": ece_improved,
            }
            for variant in ["blur1", "jpeg30"]
            for score_mode, acc_delta, auc_delta, brier_improved, ece_improved in [
                ("global", 0.0, 0.0, False, False),
                ("tile_mean", 0.002, 0.004, True, True),
                ("tile_max", 0.014, 0.016, False, False),
            ]
        ]
    ).to_csv(tiled_dino_tradeoff, index=False)
    source_stress_summary = assets_dir / "ms_cocoai_to_ishu_source_holdout_model_selection_source_summary.csv"
    pd.DataFrame(
        [
            {
                "selection_policy": "source_holdout_mean_utility_cap_0p48",
                "heldout_source_name": "sd3",
                "source_holdout_utility_mean": 1.42,
                "source_holdout_recall_mean": 0.7961,
                "source_holdout_fake_miss_rate_mean": 0.2039,
                "source_holdout_predicted_positive_rate_mean": 0.13,
            },
            {
                "selection_policy": "source_holdout_mean_utility_cap_0p48",
                "heldout_source_name": "sdxl",
                "source_holdout_utility_mean": 1.89,
                "source_holdout_recall_mean": 0.98,
                "source_holdout_fake_miss_rate_mean": 0.02,
                "source_holdout_predicted_positive_rate_mean": 0.21,
            },
        ]
    ).to_csv(source_stress_summary, index=False)

    for relative in [
        "reports/assets/publication_score_fusion_clip_frontier.png",
        "reports/assets/publication_triage_operating_points.png",
        "reports/assets/publication_reverse_operating_points.png",
        "reports/assets/publication_reverse_transform_robustness.png",
        "reports/assets/qualitative_seed17_scp_fusion_false_negatives.png",
        "reports/assets/qualitative_seed29_scp_fusion_false_negatives.png",
    ]:
        _write_image(repo_root / relative, (160, 100))

    subprocess.run(
        [
            sys.executable,
            str(scripts_dir / "build_source_stress_figure.py"),
            "--source-summary",
            "reports/assets/ms_cocoai_to_ishu_source_holdout_model_selection_source_summary.csv",
            "--out-dir",
            "reports/assets",
            "--report-out",
            "reports/source_holdout_generator_stress_2026_06_14.md",
            "--dpi",
            "90",
        ],
        cwd=repo_root,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(scripts_dir / "build_dfrws_poster_brief.py"),
            "--core-results",
            "reports/assets/publication_core_results.csv",
            "--claim-matrix",
            "reports/assets/claim_evidence_matrix.csv",
            "--tiled-dino-tradeoff",
            "reports/assets/tiled_dinov2_calibration_tradeoff.csv",
            "--out-path",
            "reports/dfrws_poster_brief_2026_06_13.md",
            "--key-numbers-out",
            "reports/assets/dfrws_poster_key_numbers.csv",
        ],
        cwd=repo_root,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(scripts_dir / "build_dfrws_poster_figures.py"),
            "--core-results",
            "reports/assets/publication_core_results.csv",
            "--out-dir",
            "reports/assets",
            "--report-out",
            "reports/dfrws_poster_native_figures_2026_06_13.md",
            "--dpi",
            "90",
        ],
        cwd=repo_root,
        check=True,
    )

    (reports_dir / "dfrws_poster_draft_v2_2026_06_13.md").write_text("poster draft\n", encoding="utf-8")
    _write_image(assets_dir / "dfrws_poster_draft_v2_2026_06_13.png", (320, 180))
    (assets_dir / "dfrws_poster_draft_v2_2026_06_13.pptx").write_bytes(b"x" * 12000)

    out_path = tmp_path / "dfrws_poster_package_lint.md"
    checks_out = tmp_path / "dfrws_poster_package_lint.csv"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "lint_dfrws_poster_package.py"),
            "--repo-root",
            str(repo_root),
            "--core-results",
            str(core_results),
            "--claim-matrix",
            str(claim_matrix),
            "--brief",
            str(reports_dir / "dfrws_poster_brief_2026_06_13.md"),
            "--key-numbers",
            str(assets_dir / "dfrws_poster_key_numbers.csv"),
            "--transfer-panel",
            str(assets_dir / "dfrws_poster_transfer_panel.csv"),
            "--robustness-panel",
            str(assets_dir / "dfrws_poster_robustness_panel.csv"),
            "--tiled-dino-tradeoff",
            str(tiled_dino_tradeoff),
            "--min-panel-width",
            "900",
            "--min-panel-height",
            "350",
            "--min-preview-width",
            "300",
            "--min-preview-height",
            "150",
            "--out-path",
            str(out_path),
            "--checks-out",
            str(checks_out),
        ],
        cwd=ROOT,
        check=True,
    )

    report = out_path.read_text(encoding="utf-8")
    checks = pd.read_csv(checks_out)
    assert "Status: **PASS**" in report
    assert checks["passed"].all()
    assert "poster key numbers match canonical results" in checks["check"].tolist()
    assert "transfer panel CSV matches canonical results" in checks["check"].tolist()
    assert "brief tiled-DINO phrase present: tile_max" in checks["check"].tolist()
    assert "brief figure package files exist" in checks["check"].tolist()
    assert "poster draft PPTX is nontrivial" in checks["check"].tolist()


def _write_image(path: Path, size: tuple[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(path)
