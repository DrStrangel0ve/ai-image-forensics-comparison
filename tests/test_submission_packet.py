from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_submission_packet_builder_writes_manifest_and_validates_artifacts(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    for artifact in _load_artifacts():
        path = repo_root / artifact["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("placeholder\n", encoding="utf-8")

    core_results = tmp_path / "publication_core_results.csv"
    claim_matrix = tmp_path / "claim_evidence_matrix.csv"
    out_path = tmp_path / "submission_packet.md"
    manifest_out = tmp_path / "submission_packet_manifest.csv"

    pd.DataFrame(
        {
            "finding_id": [
                "ishu_same_combined_v3",
                "ishu_same_resnet18",
                "ishu_same_physics_guided",
                "ishu_to_ms_clip_standalone",
                "ishu_to_ms_scp_fusion_all_foundation",
                "ishu_to_ms_triage5_clip_standalone",
                "ms_to_ishu_tuned_fusion_constraint_sweep_best",
                "ms_to_ishu_tuned_fusion_native_tiling_best",
                "ms_to_ishu_tuned_fusion_jpeg30",
                "ms_to_ishu_tuned_fusion_blur1",
                "ms_to_ishu_tuned_fusion_social_720p",
            ],
            "method": ["method"] * 11,
            "setting": ["setting"] * 11,
            "accuracy": [0.70] * 11,
            "auc": [0.80] * 11,
            "brier": [0.20] * 11,
            "ece": [0.10] * 11,
            "predicted_fake_rate": [0.50] * 11,
            "coverage": [pd.NA] * 11,
            "decided_accuracy": [pd.NA] * 11,
            "source": ["source"] * 11,
            "interpretation": ["interpretation"] * 11,
        }
    ).to_csv(core_results, index=False)

    pd.DataFrame(
        {
            "claim_id": ["clip_transfer_frontier", "scp_fusion_is_diagnostic"],
            "claim": ["claim", "claim"],
            "submission_use": ["DFRWS headline", "DFF method framing"],
            "status": ["ready", "ready_with_caveat"],
            "evidence_finding_ids": ["ishu_to_ms_clip_standalone", "ms_to_ishu_tuned_fusion_native_tiling_best"],
            "evidence_summary": ["summary", "summary"],
            "primary_artifact": ["artifact", "artifact"],
            "risk_or_caveat": ["caveat", "caveat"],
            "next_action": ["next", "next"],
        }
    ).to_csv(claim_matrix, index=False)

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_submission_packet.py"),
            "--repo-root",
            str(repo_root),
            "--core-results",
            str(core_results),
            "--claim-matrix",
            str(claim_matrix),
            "--out-path",
            str(out_path),
            "--manifest-out",
            str(manifest_out),
        ],
        cwd=ROOT,
        check=True,
    )

    text = out_path.read_text(encoding="utf-8")
    manifest = pd.read_csv(manifest_out)

    assert "Submission Packet Manifest" in text
    assert "DFRWS Packet" in text
    assert "WIFS Packet" in text
    assert "DFF Packet" in text
    assert "ms_to_ishu_tuned_fusion_native_tiling_best" in text
    assert "classic multi-light photometric stereo" in text
    assert "publication control suite dry run" in text
    assert "opportunity watchlist" in text
    assert "external benchmark readiness" in text
    assert "external benchmark claim lint" in text
    assert "SOTA gap report" in text
    assert "SOTA gap closure plan" in text
    assert "Do not claim SOTA" in text
    assert "robustness failure ranking" in text
    assert "tiled foundation comparison" in text
    assert "tiled foundation fusion comparison" in text
    assert manifest["exists"].all()
    assert "reports/assets/dfrws_poster_robustness_panel.png" in set(manifest["path"])
    assert "reports/publication_control_suite_2026_06_14.md" in set(manifest["path"])
    assert "reports/opportunity_watchlist_2026_06_14.md" in set(manifest["path"])
    assert "reports/external_benchmark_readiness_2026_06_14.md" in set(manifest["path"])
    assert "reports/external_benchmark_claim_lint_2026_06_14.md" in set(manifest["path"])
    assert "reports/sota_gap_report_2026_06_14.md" in set(manifest["path"])
    assert "reports/assets/sota_gap_report.csv" in set(manifest["path"])
    assert "reports/sota_gap_closure_plan_2026_06_14.md" in set(manifest["path"])
    assert "reports/assets/sota_gap_closure_plan.csv" in set(manifest["path"])
    assert "reports/robustness_failure_ranking_2026_06_14.md" in set(manifest["path"])
    assert "reports/tiled_clip_reverse_transfer_2026_06_14.md" in set(manifest["path"])
    assert "reports/assets/tiled_clip_reverse_transfer_summary.csv" in set(manifest["path"])
    assert "reports/tiled_dinov2_reverse_transfer_2026_06_14.md" in set(manifest["path"])
    assert "reports/assets/tiled_dinov2_reverse_transfer_summary.csv" in set(manifest["path"])
    assert "reports/tiled_convnext_reverse_transfer_2026_06_14.md" in set(manifest["path"])
    assert "reports/assets/tiled_convnext_reverse_transfer_summary.csv" in set(manifest["path"])
    assert "reports/tiled_foundation_reverse_transfer_comparison_2026_06_14.md" in set(manifest["path"])
    assert "reports/assets/tiled_foundation_reverse_transfer_comparison.csv" in set(manifest["path"])
    assert "reports/ms_cocoai_to_ishu_tuned_fusion_tiled_clip_2026_06_14.md" in set(manifest["path"])
    assert "reports/ms_cocoai_to_ishu_tuned_fusion_tiled_dinov2_2026_06_14.md" in set(manifest["path"])
    assert "reports/ms_cocoai_to_ishu_tuned_fusion_tiled_convnext_2026_06_14.md" in set(manifest["path"])
    assert "reports/ms_cocoai_to_ishu_tiled_foundation_fusion_comparison_2026_06_14.md" in set(manifest["path"])
    assert "reports/assets/ms_cocoai_to_ishu_tiled_foundation_fusion_comparison.csv" in set(manifest["path"])
    assert "DFF" in manifest["venues"].str.cat(sep=",")


def _load_artifacts() -> list[dict[str, object]]:
    import importlib.util

    module_path = ROOT / "scripts" / "build_submission_packet.py"
    spec = importlib.util.spec_from_file_location("build_submission_packet", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return list(module.ARTIFACTS)
