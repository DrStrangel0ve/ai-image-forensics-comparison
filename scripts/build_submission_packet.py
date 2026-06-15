from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import pandas as pd


DEFAULT_RUN_DATE = date.today()

LEAD_FINDINGS = [
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
]

ARTIFACTS = [
    {
        "path": "README.md",
        "type": "repo",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Public project entry point with setup, dataset catalog, commands, and result map.",
        "required": True,
    },
    {
        "path": "CITATION.cff",
        "type": "repo",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Citation metadata for public release and reproducibility review.",
        "required": True,
    },
    {
        "path": "reports/reproducibility_checklist_2026_06_12.md",
        "type": "reproducibility",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Reviewer-facing map of included artifacts, external datasets, and commands.",
        "required": True,
    },
    {
        "path": "reports/submission_readiness_2026_06_13.md",
        "type": "planning",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Current readiness snapshot, lead results, figure package, and remaining gaps.",
        "required": True,
    },
    {
        "path": "reports/submission_scorecard_2026_06_14.md",
        "type": "planning",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Venue-level readiness scorecard aggregating packet artifacts, lint reports, claim caveats, deadlines, and next actions.",
        "required": False,
    },
    {
        "path": "reports/assets/submission_scorecard.csv",
        "type": "planning",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Machine-readable venue-level readiness scorecard.",
        "required": False,
    },
    {
        "path": "reports/submission_upload_checklist_2026_06_14.md",
        "type": "planning",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Venue upload checklist separating ready assets from final export, writing, and decision tasks.",
        "required": False,
    },
    {
        "path": "reports/assets/submission_upload_checklist.csv",
        "type": "planning",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Machine-readable venue upload checklist.",
        "required": False,
    },
    {
        "path": "reports/submission_critical_path_2026_06_14.md",
        "type": "planning",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Deadline-driven remaining-work queue for final writing and upload export tasks.",
        "required": False,
    },
    {
        "path": "reports/assets/submission_critical_path.csv",
        "type": "planning",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Machine-readable critical path for pending writing and final-export tasks.",
        "required": False,
    },
    {
        "path": "reports/publication_control_suite_2026_06_14.md",
        "type": "planning",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Ordered dry-run command plan for regenerating publication artifacts and lints.",
        "required": False,
    },
    {
        "path": "reports/assets/publication_control_suite.csv",
        "type": "planning",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Machine-readable publication-control command plan.",
        "required": False,
    },
    {
        "path": "reports/opportunity_watchlist_2026_06_14.md",
        "type": "planning",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Current venue/challenge watchlist separating active submission targets from closed benchmark tracks.",
        "required": False,
    },
    {
        "path": "reports/assets/opportunity_watchlist.csv",
        "type": "planning",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Machine-readable opportunity and benchmark watchlist.",
        "required": False,
    },
    {
        "path": "reports/external_benchmark_readiness_2026_06_14.md",
        "type": "planning",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Official-vs-proxy readiness audit for NTIRE/ImageCLEF-style external benchmark claims.",
        "required": False,
    },
    {
        "path": "reports/assets/external_benchmark_readiness.csv",
        "type": "planning",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Machine-readable official benchmark readiness status.",
        "required": False,
    },
    {
        "path": "reports/assets/external_benchmark_proxy_metrics.csv",
        "type": "planning",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Machine-readable proxy evidence rows for external benchmark claims.",
        "required": False,
    },
    {
        "path": "reports/external_benchmark_claim_lint_2026_06_14.md",
        "type": "quality-control",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Generated lint report checking that NTIRE/ImageCLEF proxy evidence is not described as official challenge performance.",
        "required": False,
    },
    {
        "path": "reports/assets/external_benchmark_claim_lint.csv",
        "type": "quality-control",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Machine-readable external benchmark claim lint checks.",
        "required": False,
    },
    {
        "path": "reports/sota_gap_report_2026_06_14.md",
        "type": "planning",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Guarded comparison between checked-in local proxy results and official/current SOTA benchmark anchors.",
        "required": False,
    },
    {
        "path": "reports/assets/sota_gap_report.csv",
        "type": "planning",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Machine-readable NTIRE-style SOTA gap rows for local proxy metrics.",
        "required": False,
    },
    {
        "path": "reports/sota_gap_closure_plan_2026_06_14.md",
        "type": "planning",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Prioritized experiment plan for turning the current proxy SOTA-gap evidence into official or paper-compatible benchmark evidence.",
        "required": False,
    },
    {
        "path": "reports/assets/sota_gap_closure_plan.csv",
        "type": "planning",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Machine-readable SOTA gap closure tasks, statuses, commands, and exit criteria.",
        "required": False,
    },
    {
        "path": "reports/competition_submission_dry_run_2026_06_15.md",
        "type": "quality-control",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Challenge-style dry-run proving that checked-in per-image scores can be packaged, stripped of labels, and linted before upload.",
        "required": False,
    },
    {
        "path": "reports/assets/competition_dry_run/predictions.csv",
        "type": "quality-control",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Small derived prediction fixture used to exercise competition packaging; not an upload file.",
        "required": False,
    },
    {
        "path": "reports/assets/competition_dry_run/expected_ids.csv",
        "type": "quality-control",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Expected image-id list used by the competition dry-run lint check.",
        "required": False,
    },
    {
        "path": "reports/assets/competition_dry_run/submission.csv",
        "type": "quality-control",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Upload-shaped competition dry-run submission with scores, labels, confidence, and triage but no ground-truth labels.",
        "required": False,
    },
    {
        "path": "reports/assets/competition_dry_run/submission_manifest.json",
        "type": "quality-control",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Manifest emitted by the competition submission packager for the dry-run submission.",
        "required": False,
    },
    {
        "path": "reports/assets/competition_dry_run/submission_lint.json",
        "type": "quality-control",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Machine-readable lint report proving the dry-run submission has no leakage, duplicate IDs, score errors, or ID coverage gaps.",
        "required": False,
    },
    {
        "path": "reports/submission_artifact_hashes_2026_06_15.md",
        "type": "reproducibility",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Reviewer-facing SHA-256 hash summary for checked-in submission packet artifacts.",
        "required": False,
    },
    {
        "path": "reports/assets/submission_artifact_hashes.csv",
        "type": "reproducibility",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Machine-readable SHA-256 hash manifest for checked-in submission packet artifacts.",
        "required": False,
    },
    {
        "path": "reports/submission_artifact_hashes_lint_2026_06_15.md",
        "type": "quality-control",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Generated lint report proving checked artifact hashes still match the files on disk.",
        "required": False,
    },
    {
        "path": "reports/assets/submission_artifact_hashes_lint.csv",
        "type": "quality-control",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Machine-readable lint checks for the submission artifact hash manifest.",
        "required": False,
    },
    {
        "path": "reports/submission_path_sanitization_2026_06_15.md",
        "type": "quality-control",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Public-release path sanitization report for replacing workstation-specific repository prefixes with <repo> placeholders.",
        "required": False,
    },
    {
        "path": "reports/assets/submission_path_sanitization.csv",
        "type": "quality-control",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Machine-readable path sanitization changes for checked-in text artifacts.",
        "required": False,
    },
    {
        "path": "reports/submission_privacy_audit_2026_06_15.md",
        "type": "quality-control",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Public-release audit for local absolute paths, placeholder paths, and obvious secret-like strings in packet artifacts.",
        "required": False,
    },
    {
        "path": "reports/assets/submission_privacy_audit.csv",
        "type": "quality-control",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Machine-readable privacy/local-path audit findings for checked-in submission packet artifacts.",
        "required": False,
    },
    {
        "path": "reports/reconstruction_lite_feature_set_2026_06_14.md",
        "type": "method",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Standalone AEROBLADE-lite resize-reconstruction feature ablation manifest and run command.",
        "required": False,
    },
    {
        "path": "reports/assets/reconstruction_lite_feature_manifest.csv",
        "type": "method",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Machine-readable reconstruction_lite feature manifest.",
        "required": False,
    },
    {
        "path": "reports/reconstruction_v2_feature_set_2026_06_14.md",
        "type": "method",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Extended deterministic reconstruction residual feature ablation with FFT and low-rank SVD reconstructions.",
        "required": False,
    },
    {
        "path": "reports/assets/reconstruction_v2_feature_manifest.csv",
        "type": "method",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Machine-readable reconstruction_v2 feature manifest.",
        "required": False,
    },
    {
        "path": "reports/reconstruction_v2_probe_2026_06_14.md",
        "type": "result-note",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Bounded same-domain and Ishu to MS COCOAI reconstruction_v2 diagnostic.",
        "required": False,
    },
    {
        "path": "reports/assets/reconstruction_v2_probe_seed_summary.csv",
        "type": "result-data",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Seed-level reconstruction_v2, reconstruction_lite, and combined_v3 probe metrics.",
        "required": False,
    },
    {
        "path": "reports/assets/reconstruction_v2_probe_mean_summary.csv",
        "type": "result-data",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Mean reconstruction_v2, reconstruction_lite, and combined_v3 probe metrics.",
        "required": False,
    },
    {
        "path": "reports/assets/reconstruction_v2_probe_delta_summary.csv",
        "type": "result-data",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Delta summary for reconstruction_v2 against reconstruction_lite and combined_v3.",
        "required": False,
    },
    {
        "path": "reports/reconstruction_lite_probe_2026_06_14.md",
        "type": "result-note",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Bounded three-seed reconstruction_lite vs combined_v3 ablation summary.",
        "required": False,
    },
    {
        "path": "reports/assets/reconstruction_lite_probe_seed_summary.csv",
        "type": "result-data",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Seed-level reconstruction_lite vs combined_v3 bounded probe metrics.",
        "required": False,
    },
    {
        "path": "reports/assets/reconstruction_lite_probe_mean_summary.csv",
        "type": "result-data",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Mean reconstruction_lite vs combined_v3 bounded probe metrics.",
        "required": False,
    },
    {
        "path": "reports/assets/reconstruction_lite_probe_delta_summary.csv",
        "type": "result-data",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Delta summary for reconstruction_lite against combined_v3 on the bounded probe.",
        "required": False,
    },
    {
        "path": "reports/reconstruction_lite_transfer_probe_2026_06_14.md",
        "type": "result-note",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Bounded Ishu to MS COCOAI reconstruction_lite transfer diagnostic.",
        "required": False,
    },
    {
        "path": "reports/assets/reconstruction_lite_transfer_probe_seed_summary.csv",
        "type": "result-data",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Seed-level transfer metrics for reconstruction_lite and combined_v3 bounded probes.",
        "required": False,
    },
    {
        "path": "reports/assets/reconstruction_lite_transfer_probe_mean_summary.csv",
        "type": "result-data",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Mean transfer metrics for reconstruction_lite and combined_v3 bounded probes.",
        "required": False,
    },
    {
        "path": "reports/assets/reconstruction_lite_transfer_probe_delta_summary.csv",
        "type": "result-data",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Transfer delta summary for reconstruction_lite against combined_v3.",
        "required": False,
    },
    {
        "path": "reports/reconstruction_lite_fusion_probe_2026_06_14.md",
        "type": "result-note",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Bounded Ishu to MS COCOAI reconstruction_lite plus combined_v3 score-fusion diagnostic.",
        "required": False,
    },
    {
        "path": "reports/assets/reconstruction_lite_fusion_probe_seed_summary.csv",
        "type": "result-data",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Seed-level transfer metrics for reconstruction_lite, combined_v3, and bounded score-fusion candidates.",
        "required": False,
    },
    {
        "path": "reports/assets/reconstruction_lite_fusion_probe_mean_summary.csv",
        "type": "result-data",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Mean transfer metrics for reconstruction_lite, combined_v3, and bounded score-fusion candidates.",
        "required": False,
    },
    {
        "path": "reports/assets/reconstruction_lite_fusion_probe_delta_summary.csv",
        "type": "result-data",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Transfer delta summary for bounded reconstruction_lite fusion candidates against combined_v3.",
        "required": False,
    },
    {
        "path": "reports/assets/reconstruction_lite_fusion_probe_coefficients.csv",
        "type": "result-data",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Source-fitted two-branch fusion coefficients for the bounded reconstruction_lite diagnostic.",
        "required": False,
    },
    {
        "path": "reports/literature_map_2026_06_14.md",
        "type": "literature",
        "venues": "WIFS,DFF",
        "purpose": "Paper-facing map from related-work references to method claims, paper sections, and caveats.",
        "required": True,
    },
    {
        "path": "reports/assets/literature_map.csv",
        "type": "literature",
        "venues": "WIFS,DFF",
        "purpose": "Machine-readable literature map for WIFS/DFF related-work drafting.",
        "required": True,
    },
    {
        "path": "references.bib",
        "type": "bibliography",
        "venues": "WIFS,DFF",
        "purpose": "Draft BibTeX generated from the literature map for WIFS/DFF skeleton citations.",
        "required": True,
    },
    {
        "path": "reports/references_bib_2026_06_14.md",
        "type": "bibliography",
        "venues": "WIFS,DFF",
        "purpose": "Report listing draft BibTeX entries and metadata-verification caveats.",
        "required": True,
    },
    {
        "path": "reports/assets/references_bib_manifest.csv",
        "type": "bibliography",
        "venues": "WIFS,DFF",
        "purpose": "Machine-readable manifest for draft BibTeX entries.",
        "required": True,
    },
    {
        "path": "reports/assets/publication_core_results.csv",
        "type": "table",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Canonical machine-readable result rows used by claim and figure builders.",
        "required": True,
    },
    {
        "path": "reports/assets/publication_core_results.md",
        "type": "table",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Readable core result table for paper drafting.",
        "required": True,
    },
    {
        "path": "reports/ms_cocoai_to_ishu_source_holdout_model_selection_2026_06_13.md",
        "type": "result-note",
        "venues": "WIFS,DFF",
        "purpose": "Reverse-transfer source-heldout model-selection report with per-generator stress diagnostics.",
        "required": False,
    },
    {
        "path": "reports/assets/ms_cocoai_to_ishu_source_holdout_model_selection_source_summary.csv",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "Machine-readable held-out-generator stress summary for selected source-holdout policies.",
        "required": False,
    },
    {
        "path": "reports/source_holdout_generator_stress_2026_06_14.md",
        "type": "result-note",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Readable held-out-generator stress figure note for the capped source-holdout policy.",
        "required": False,
    },
    {
        "path": "reports/assets/source_holdout_generator_stress.csv",
        "type": "table",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Machine-readable source-stress figure data sorted by held-out generator utility.",
        "required": False,
    },
    {
        "path": "reports/assets/source_holdout_generator_stress.png",
        "type": "figure",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Held-out-generator stress chart highlighting the weakest source family.",
        "required": False,
    },
    {
        "path": "reports/assets/source_holdout_generator_stress.svg",
        "type": "figure",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Editable SVG source for the held-out-generator stress chart.",
        "required": False,
    },
    {
        "path": "reports/submission_result_tables_2026_06_14.md",
        "type": "table",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Copy-ready compact result tables for poster and paper drafting.",
        "required": True,
    },
    {
        "path": "reports/assets/submission_result_table_manifest.csv",
        "type": "table",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Machine-readable manifest for compact submission result tables.",
        "required": True,
    },
    {
        "path": "reports/assets/submission_table_same_domain_anchor.csv",
        "type": "table",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Compact same-domain anchor table.",
        "required": True,
    },
    {
        "path": "reports/assets/submission_table_transfer_frontier.csv",
        "type": "table",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Compact Ishu-to-MS-COCOAI transfer frontier table.",
        "required": True,
    },
    {
        "path": "reports/assets/submission_table_reverse_operating_points.csv",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "Compact MS-COCOAI-to-Ishu reverse operating point table.",
        "required": True,
    },
    {
        "path": "reports/assets/submission_table_robustness_stress.csv",
        "type": "table",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Compact reverse tuned-fusion robustness stress table with clean-baseline deltas.",
        "required": True,
    },
    {
        "path": "reports/assets/submission_table_source_holdout_stress.csv",
        "type": "table",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Compact held-out-generator stress table for the capped source-holdout policy.",
        "required": True,
    },
    {
        "path": "reports/assets/submission_table_reconstruction_ablation.csv",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "Compact reconstruction residual ablation table showing same-domain gain and transfer source sensitivity.",
        "required": True,
    },
    {
        "path": "reports/submission_result_tables_lint_2026_06_14.md",
        "type": "quality-control",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Generated lint report proving compact result tables match canonical metrics, robustness deltas, and claim-evidence IDs.",
        "required": False,
    },
    {
        "path": "reports/assets/submission_result_tables_lint.csv",
        "type": "quality-control",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Machine-readable compact result-table consistency checks.",
        "required": False,
    },
    {
        "path": "reports/submission_latex_tables_2026_06_14.md",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "Report listing generated LaTeX table fragments for paper drafting.",
        "required": True,
    },
    {
        "path": "reports/assets/latex_tables/submission_latex_table_manifest.csv",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "Machine-readable manifest for generated LaTeX table fragments.",
        "required": True,
    },
    {
        "path": "reports/assets/latex_tables/same_domain_anchor.tex",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "LaTeX fragment for the same-domain anchor table.",
        "required": True,
    },
    {
        "path": "reports/assets/latex_tables/transfer_frontier.tex",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "LaTeX fragment for the Ishu-to-MS-COCOAI transfer frontier table.",
        "required": True,
    },
    {
        "path": "reports/assets/latex_tables/reverse_operating_points.tex",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "LaTeX fragment for the reverse operating point table.",
        "required": True,
    },
    {
        "path": "reports/assets/latex_tables/robustness_stress.tex",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "LaTeX fragment for the robustness stress table.",
        "required": True,
    },
    {
        "path": "reports/assets/latex_tables/method_family_comparison.tex",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "LaTeX fragment for the method-family comparison table.",
        "required": True,
    },
    {
        "path": "reports/assets/latex_tables/calibration_operating_modes.tex",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "LaTeX fragment for the objective-specific calibration and operating-mode table.",
        "required": True,
    },
    {
        "path": "reports/assets/latex_tables/source_holdout_stress.tex",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "LaTeX fragment for the held-out generator stress table.",
        "required": True,
    },
    {
        "path": "reports/assets/latex_tables/reconstruction_ablation.tex",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "LaTeX fragment for the reconstruction residual ablation table.",
        "required": True,
    },
    {
        "path": "reports/submission_paper_skeletons_2026_06_14.md",
        "type": "paper-draft",
        "venues": "WIFS,DFF",
        "purpose": "Report listing generated WIFS/DFF paper skeletons.",
        "required": True,
    },
    {
        "path": "reports/assets/paper_skeletons/submission_paper_skeleton_manifest.csv",
        "type": "paper-draft",
        "venues": "WIFS,DFF",
        "purpose": "Machine-readable manifest for generated WIFS/DFF paper skeletons.",
        "required": True,
    },
    {
        "path": "reports/assets/paper_skeletons/wifs_2026_paper_skeleton.tex",
        "type": "paper-draft",
        "venues": "WIFS",
        "purpose": "IEEEtran-style WIFS paper skeleton with generated abstract, tables, and figures.",
        "required": True,
    },
    {
        "path": "reports/assets/paper_skeletons/dff_2026_workshop_skeleton.tex",
        "type": "paper-draft",
        "venues": "DFF",
        "purpose": "ACM-style DFF workshop skeleton with generated abstract, tables, and figures.",
        "required": True,
    },
    {
        "path": "reports/paper_skeleton_lint_2026_06_14.md",
        "type": "quality-control",
        "venues": "WIFS,DFF",
        "purpose": "Generated lint report for WIFS/DFF paper skeleton structure, referenced assets, and claim guardrails.",
        "required": False,
    },
    {
        "path": "reports/assets/paper_skeleton_lint.csv",
        "type": "quality-control",
        "venues": "WIFS,DFF",
        "purpose": "Machine-readable WIFS/DFF paper skeleton lint checks.",
        "required": False,
    },
    {
        "path": "reports/assets/claim_evidence_matrix.csv",
        "type": "claim-audit",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Machine-readable claim-to-evidence map with caveats and next actions.",
        "required": True,
    },
    {
        "path": "reports/assets/claim_evidence_matrix.md",
        "type": "claim-audit",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Readable claim-to-evidence map for authoring and review.",
        "required": True,
    },
    {
        "path": "reports/claim_evidence_matrix_lint_2026_06_14.md",
        "type": "quality-control",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Generated lint report checking claim IDs, artifacts, evidence IDs, caveats, and overclaim language.",
        "required": False,
    },
    {
        "path": "reports/assets/claim_evidence_matrix_lint.csv",
        "type": "quality-control",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Machine-readable claim-evidence matrix lint checks.",
        "required": False,
    },
    {
        "path": "reports/method_family_comparison_2026_06_14.md",
        "type": "analysis",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Paper-facing method-family comparison separating ranking, calibration, operating-point, and triage winners.",
        "required": False,
    },
    {
        "path": "reports/assets/method_family_comparison.csv",
        "type": "analysis",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Machine-readable method-family winner table by scenario and criterion.",
        "required": False,
    },
    {
        "path": "reports/calibration_operating_modes_2026_06_14.md",
        "type": "analysis",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Paper-facing calibration operating-mode synthesis for ranking, decision, Brier, ECE, and tiled-DINO mode choices.",
        "required": False,
    },
    {
        "path": "reports/assets/calibration_operating_modes.csv",
        "type": "analysis",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Machine-readable calibration operating-mode selector table.",
        "required": False,
    },
    {
        "path": "reports/submission_text_drafts_2026_06_14.md",
        "type": "writing",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Venue-specific abstracts, contribution wording, result sentences, and paper skeletons.",
        "required": True,
    },
    {
        "path": "reports/assets/submission_text_drafts_word_counts.csv",
        "type": "writing",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Machine-readable word counts for generated DFRWS, WIFS, and DFF abstracts.",
        "required": True,
    },
    {
        "path": "reports/paper_section_drafts_2026_06_14.md",
        "type": "writing",
        "venues": "WIFS,DFF",
        "purpose": "Generated WIFS/DFF prose section drafts tied to current metrics, literature, and caveats.",
        "required": True,
    },
    {
        "path": "reports/assets/paper_section_draft_manifest.csv",
        "type": "writing",
        "venues": "WIFS,DFF",
        "purpose": "Machine-readable section word counts and caveat/metric flags for generated paper prose drafts.",
        "required": True,
    },
    {
        "path": "reports/paper_section_drafts_lint_2026_06_14.md",
        "type": "quality-control",
        "venues": "WIFS,DFF",
        "purpose": "Generated lint report for WIFS/DFF section prose, required metric anchors, caveats, and overclaim checks.",
        "required": False,
    },
    {
        "path": "reports/assets/paper_section_drafts_lint.csv",
        "type": "quality-control",
        "venues": "WIFS,DFF",
        "purpose": "Machine-readable WIFS/DFF section prose lint checks.",
        "required": False,
    },
    {
        "path": "reports/submission_package_lint_2026_06_14.md",
        "type": "quality-control",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Generated lint report for missing packet assets, abstract word counts, and overclaim language.",
        "required": False,
    },
    {
        "path": "reports/assets/submission_package_lint.csv",
        "type": "quality-control",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Machine-readable submission package lint checks.",
        "required": False,
    },
    {
        "path": "reports/dfrws_poster_brief_2026_06_13.md",
        "type": "poster",
        "venues": "DFRWS",
        "purpose": "Poster spine, key numbers, figure package, and overclaim warnings.",
        "required": True,
    },
    {
        "path": "reports/assets/dfrws_poster_key_numbers.csv",
        "type": "poster",
        "venues": "DFRWS",
        "purpose": "Compact poster-leading key number table.",
        "required": True,
    },
    {
        "path": "reports/dfrws_poster_native_figures_2026_06_13.md",
        "type": "poster",
        "venues": "DFRWS",
        "purpose": "Large-label poster figure pack with transfer and robustness panels.",
        "required": True,
    },
    {
        "path": "reports/assets/dfrws_poster_transfer_panel.png",
        "type": "figure",
        "venues": "DFRWS",
        "purpose": "Poster-native transfer headline panel.",
        "required": True,
    },
    {
        "path": "reports/assets/dfrws_poster_transfer_panel.svg",
        "type": "figure",
        "venues": "DFRWS",
        "purpose": "Editable SVG source for the poster transfer panel.",
        "required": True,
    },
    {
        "path": "reports/assets/dfrws_poster_robustness_panel.png",
        "type": "figure",
        "venues": "DFRWS",
        "purpose": "Poster-native robustness panel with native-tiling diagnostic and transforms.",
        "required": True,
    },
    {
        "path": "reports/assets/dfrws_poster_robustness_panel.svg",
        "type": "figure",
        "venues": "DFRWS",
        "purpose": "Editable SVG source for the poster robustness panel.",
        "required": True,
    },
    {
        "path": "reports/dfrws_poster_draft_v2_2026_06_13.md",
        "type": "poster",
        "venues": "DFRWS",
        "purpose": "Editable poster draft notes and preview references.",
        "required": True,
    },
    {
        "path": "reports/assets/dfrws_poster_draft_v2_2026_06_13.pptx",
        "type": "poster",
        "venues": "DFRWS",
        "purpose": "Editable PowerPoint poster draft.",
        "required": True,
    },
    {
        "path": "reports/assets/dfrws_poster_draft_v2_2026_06_13.png",
        "type": "poster",
        "venues": "DFRWS",
        "purpose": "Poster draft PNG preview.",
        "required": True,
    },
    {
        "path": "reports/dfrws_poster_package_lint_2026_06_14.md",
        "type": "quality-control",
        "venues": "DFRWS",
        "purpose": "Generated lint report for DFRWS poster assets, key-number consistency, figure dimensions, and overclaim cautions.",
        "required": False,
    },
    {
        "path": "reports/assets/dfrws_poster_package_lint.csv",
        "type": "quality-control",
        "venues": "DFRWS",
        "purpose": "Machine-readable DFRWS poster package lint checks.",
        "required": False,
    },
    {
        "path": "reports/assets/publication_score_fusion_clip_frontier.png",
        "type": "figure",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "CLIP transfer frontier and all-foundation fusion comparison.",
        "required": True,
    },
    {
        "path": "reports/assets/publication_triage_operating_points.png",
        "type": "figure",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "High-confidence source-heldout triage operating points.",
        "required": True,
    },
    {
        "path": "reports/assets/publication_reverse_operating_points.png",
        "type": "figure",
        "venues": "WIFS,DFF",
        "purpose": "Reverse-transfer source-threshold and operating-point comparison.",
        "required": True,
    },
    {
        "path": "reports/assets/publication_reverse_transform_robustness.png",
        "type": "figure",
        "venues": "WIFS,DFF",
        "purpose": "Reverse tuned-fusion target-transform robustness stress result.",
        "required": True,
    },
    {
        "path": "reports/assets/publication_reverse_fusion_tradeoff.png",
        "type": "figure",
        "venues": "WIFS,DFF",
        "purpose": "Reverse fusion ranking, calibration, and operating-point tradeoff.",
        "required": True,
    },
    {
        "path": "reports/assets/publication_score_fusion_dinov2_gain.png",
        "type": "figure",
        "venues": "WIFS,DFF",
        "purpose": "DINOv2 branch-complementarity and fusion gain figure.",
        "required": True,
    },
    {
        "path": "reports/assets/publication_source_heldout_calibration.png",
        "type": "figure",
        "venues": "WIFS,DFF",
        "purpose": "Source-heldout calibration evidence.",
        "required": True,
    },
    {
        "path": "reports/assets/qualitative_seed17_scp_fusion_false_negatives.png",
        "type": "qualitative",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Appendix or backup failure-case grid for generated-image misses.",
        "required": True,
    },
    {
        "path": "reports/assets/qualitative_seed29_scp_fusion_false_negatives.png",
        "type": "qualitative",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Selected DFRWS poster failure-case grid and second-seed repeatability grid.",
        "required": True,
    },
    {
        "path": "reports/dfrws_qualitative_grid_selection_2026_06_14.md",
        "type": "planning",
        "venues": "DFRWS,DFF",
        "purpose": "Generated selection report choosing the seed-29 false-negative grid for the DFRWS poster.",
        "required": True,
    },
    {
        "path": "reports/assets/dfrws_qualitative_grid_selection.csv",
        "type": "planning",
        "venues": "DFRWS,DFF",
        "purpose": "Machine-readable DFRWS qualitative-grid selection audit.",
        "required": True,
    },
    {
        "path": "reports/wifs_breadth_decision_2026_06_14.md",
        "type": "planning",
        "venues": "WIFS,DFF",
        "purpose": "Generated WIFS breadth decision freezing the 6-page paper scope and deferring high-risk extra experiments.",
        "required": True,
    },
    {
        "path": "reports/assets/wifs_breadth_decision.csv",
        "type": "planning",
        "venues": "WIFS,DFF",
        "purpose": "Machine-readable WIFS breadth option ranking.",
        "required": True,
    },
    {
        "path": "reports/manuscript_assembly_map_2026_06_14.md",
        "type": "planning",
        "venues": "WIFS,DFF",
        "purpose": "Generated WIFS/DFF manuscript assembly map tying section drafts, page budgets, figures, and tables to writing tasks.",
        "required": True,
    },
    {
        "path": "reports/assets/manuscript_assembly_map.csv",
        "type": "planning",
        "venues": "WIFS,DFF",
        "purpose": "Machine-readable manuscript assembly map for WIFS and DFF writing.",
        "required": True,
    },
    {
        "path": "reports/wifs_manuscript_draft_2026_06_14.md",
        "type": "paper-draft",
        "venues": "WIFS",
        "purpose": "Assembled WIFS markdown manuscript draft with section prose and asset callouts.",
        "required": True,
    },
    {
        "path": "reports/dff_manuscript_draft_2026_06_14.md",
        "type": "paper-draft",
        "venues": "DFF",
        "purpose": "Assembled DFF markdown manuscript draft with section prose and asset callouts.",
        "required": True,
    },
    {
        "path": "reports/assets/manuscript_draft_manifest.csv",
        "type": "paper-draft",
        "venues": "WIFS,DFF",
        "purpose": "Machine-readable WIFS/DFF manuscript draft manifest.",
        "required": True,
    },
    {
        "path": "reports/manuscript_drafts_lint_2026_06_14.md",
        "type": "quality-control",
        "venues": "WIFS,DFF",
        "purpose": "Generated lint report checking WIFS/DFF manuscript draft sections, asset callouts, word counts, and overclaim guardrails.",
        "required": False,
    },
    {
        "path": "reports/assets/manuscript_drafts_lint.csv",
        "type": "quality-control",
        "venues": "WIFS,DFF",
        "purpose": "Machine-readable manuscript draft lint checks.",
        "required": False,
    },
    {
        "path": "reports/physics_guided_vs_resnet_2026_06_12.md",
        "type": "result-note",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Conservative physics-guided ResNet versus vanilla ResNet framing.",
        "required": True,
    },
    {
        "path": "reports/combined_v4_full_transfer_summary_2026_06_13.md",
        "type": "result-note",
        "venues": "WIFS,DFF",
        "purpose": "combined_v4 transfer gate and ablation caveat.",
        "required": True,
    },
    {
        "path": "reports/combined_v4_source_slice_diagnostics_2026_06_13.md",
        "type": "result-note",
        "venues": "WIFS,DFF",
        "purpose": "Generator/category slice analysis for combined_v4 caveats.",
        "required": True,
    },
    {
        "path": "reports/ms_cocoai_to_ishu_tuned_fusion_native_tiling_2026_06_13.md",
        "type": "result-note",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Bounded fused native-tiling diagnostic.",
        "required": True,
    },
    {
        "path": "reports/robustness_failure_ranking_2026_06_14.md",
        "type": "result-note",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Ranked target-transform failure summary for reverse tuned-fusion SCP-Fusion.",
        "required": False,
    },
    {
        "path": "reports/assets/robustness_failure_ranking.csv",
        "type": "table",
        "venues": "DFRWS,WIFS,DFF",
        "purpose": "Machine-readable ranked robustness stress table.",
        "required": False,
    },
    {
        "path": "reports/tiled_clip_reverse_transfer_2026_06_14.md",
        "type": "result-note",
        "venues": "WIFS,DFF",
        "purpose": "Three-seed reverse-transfer diagnostic comparing global CLIP scores with native-tile mean/max/top-2 aggregation.",
        "required": False,
    },
    {
        "path": "reports/assets/tiled_clip_reverse_transfer_summary.csv",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "Machine-readable three-seed summary for tiled CLIP reverse-transfer score modes.",
        "required": False,
    },
    {
        "path": "reports/assets/tiled_clip_reverse_transfer_seed_metrics.csv",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "Per-seed metrics for tiled CLIP reverse-transfer score modes.",
        "required": False,
    },
    {
        "path": "reports/assets/tiled_clip_reverse_transfer_detail.csv",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "Per-image scores for tiled CLIP reverse-transfer diagnostics.",
        "required": False,
    },
    {
        "path": "reports/tiled_dinov2_reverse_transfer_2026_06_14.md",
        "type": "result-note",
        "venues": "WIFS,DFF",
        "purpose": "Three-seed reverse-transfer diagnostic comparing global DINOv2 scores with native-tile mean/max/top-2 aggregation.",
        "required": False,
    },
    {
        "path": "reports/assets/tiled_dinov2_reverse_transfer_summary.csv",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "Machine-readable three-seed summary for tiled DINOv2 reverse-transfer score modes.",
        "required": False,
    },
    {
        "path": "reports/assets/tiled_dinov2_reverse_transfer_seed_metrics.csv",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "Per-seed metrics for tiled DINOv2 reverse-transfer score modes.",
        "required": False,
    },
    {
        "path": "reports/assets/tiled_dinov2_reverse_transfer_detail.csv",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "Per-image scores for tiled DINOv2 reverse-transfer diagnostics.",
        "required": False,
    },
    {
        "path": "reports/tiled_convnext_reverse_transfer_2026_06_14.md",
        "type": "result-note",
        "venues": "WIFS,DFF",
        "purpose": "Three-seed reverse-transfer diagnostic comparing global ConvNeXt scores with native-tile mean/max/top-2 aggregation.",
        "required": False,
    },
    {
        "path": "reports/assets/tiled_convnext_reverse_transfer_summary.csv",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "Machine-readable three-seed summary for tiled ConvNeXt reverse-transfer score modes.",
        "required": False,
    },
    {
        "path": "reports/assets/tiled_convnext_reverse_transfer_seed_metrics.csv",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "Per-seed metrics for tiled ConvNeXt reverse-transfer score modes.",
        "required": False,
    },
    {
        "path": "reports/assets/tiled_convnext_reverse_transfer_detail.csv",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "Per-image scores for tiled ConvNeXt reverse-transfer diagnostics.",
        "required": False,
    },
    {
        "path": "reports/tiled_foundation_reverse_transfer_comparison_2026_06_14.md",
        "type": "result-note",
        "venues": "WIFS,DFF",
        "purpose": "Cross-encoder tiled foundation comparison identifying the strongest accuracy and ranking aggregation modes.",
        "required": False,
    },
    {
        "path": "reports/assets/tiled_foundation_reverse_transfer_comparison.csv",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "Machine-readable CLIP/DINOv2/ConvNeXt tiled reverse-transfer comparison.",
        "required": False,
    },
    {
        "path": "reports/ms_cocoai_to_ishu_tuned_fusion_tiled_clip_2026_06_14.md",
        "type": "result-note",
        "venues": "WIFS,DFF",
        "purpose": "Fixed reverse SCP-Fusion diagnostic after replacing the CLIP target branch with tiled target scores.",
        "required": False,
    },
    {
        "path": "reports/assets/ms_cocoai_to_ishu_tuned_fusion_tiled_clip_summary.csv",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "Machine-readable fixed-fusion summary for tiled CLIP branch replacement.",
        "required": False,
    },
    {
        "path": "reports/assets/ms_cocoai_to_ishu_tuned_fusion_tiled_clip_detail.csv",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "Per-seed fixed-fusion detail for tiled CLIP branch replacement.",
        "required": False,
    },
    {
        "path": "reports/ms_cocoai_to_ishu_tuned_fusion_tiled_dinov2_2026_06_14.md",
        "type": "result-note",
        "venues": "WIFS,DFF",
        "purpose": "Fixed reverse SCP-Fusion diagnostic after replacing the DINOv2 target branch with tiled target scores.",
        "required": False,
    },
    {
        "path": "reports/assets/ms_cocoai_to_ishu_tuned_fusion_tiled_dinov2_summary.csv",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "Machine-readable fixed-fusion summary for tiled DINOv2 branch replacement.",
        "required": False,
    },
    {
        "path": "reports/assets/ms_cocoai_to_ishu_tuned_fusion_tiled_dinov2_detail.csv",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "Per-seed fixed-fusion detail for tiled DINOv2 branch replacement.",
        "required": False,
    },
    {
        "path": "reports/ms_cocoai_to_ishu_tuned_fusion_tiled_convnext_2026_06_14.md",
        "type": "result-note",
        "venues": "WIFS,DFF",
        "purpose": "Fixed reverse SCP-Fusion diagnostic after replacing the ConvNeXt target branch with tiled target scores.",
        "required": False,
    },
    {
        "path": "reports/assets/ms_cocoai_to_ishu_tuned_fusion_tiled_convnext_summary.csv",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "Machine-readable fixed-fusion summary for tiled ConvNeXt branch replacement.",
        "required": False,
    },
    {
        "path": "reports/assets/ms_cocoai_to_ishu_tuned_fusion_tiled_convnext_detail.csv",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "Per-seed fixed-fusion detail for tiled ConvNeXt branch replacement.",
        "required": False,
    },
    {
        "path": "reports/ms_cocoai_to_ishu_tiled_foundation_fusion_comparison_2026_06_14.md",
        "type": "result-note",
        "venues": "WIFS,DFF",
        "purpose": "Cross-branch fixed-fusion comparison for tiled foundation target replacements.",
        "required": False,
    },
    {
        "path": "reports/assets/ms_cocoai_to_ishu_tiled_foundation_fusion_comparison.csv",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "Machine-readable fixed-fusion comparison for tiled foundation target replacements.",
        "required": False,
    },
    {
        "path": "reports/tiled_dinov2_blur1_reverse_transfer_2026_06_14.md",
        "type": "result-note",
        "venues": "WIFS,DFF",
        "purpose": "Blur1 robustness probe for tiled DINOv2 reverse-transfer score aggregation.",
        "required": False,
    },
    {
        "path": "reports/assets/tiled_dinov2_blur1_reverse_transfer_summary.csv",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "Machine-readable summary for blur1 tiled DINOv2 reverse-transfer score modes.",
        "required": False,
    },
    {
        "path": "reports/assets/tiled_dinov2_blur1_reverse_transfer_seed_metrics.csv",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "Per-seed metrics for blur1 tiled DINOv2 reverse-transfer score modes.",
        "required": False,
    },
    {
        "path": "reports/assets/tiled_dinov2_blur1_reverse_transfer_detail.csv",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "Per-image blur1 tiled DINOv2 scores for robustness diagnostics.",
        "required": False,
    },
    {
        "path": "reports/ms_cocoai_to_ishu_tuned_fusion_blur1_tiled_dinov2_2026_06_14.md",
        "type": "result-note",
        "venues": "WIFS,DFF",
        "purpose": "Fixed reverse SCP-Fusion blur1 robustness diagnostic after replacing the DINOv2 target branch with tiled target scores.",
        "required": False,
    },
    {
        "path": "reports/assets/ms_cocoai_to_ishu_tuned_fusion_blur1_tiled_dinov2_summary.csv",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "Machine-readable fixed-fusion blur1 summary for tiled DINOv2 branch replacement.",
        "required": False,
    },
    {
        "path": "reports/assets/ms_cocoai_to_ishu_tuned_fusion_blur1_tiled_dinov2_detail.csv",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "Per-seed fixed-fusion blur1 detail for tiled DINOv2 branch replacement.",
        "required": False,
    },
    {
        "path": "reports/tiled_dinov2_jpeg30_reverse_transfer_2026_06_14.md",
        "type": "result-note",
        "venues": "WIFS,DFF",
        "purpose": "JPEG30 robustness probe for tiled DINOv2 reverse-transfer score aggregation.",
        "required": False,
    },
    {
        "path": "reports/assets/tiled_dinov2_jpeg30_reverse_transfer_summary.csv",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "Machine-readable summary for JPEG30 tiled DINOv2 reverse-transfer score modes.",
        "required": False,
    },
    {
        "path": "reports/assets/tiled_dinov2_jpeg30_reverse_transfer_seed_metrics.csv",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "Per-seed metrics for JPEG30 tiled DINOv2 reverse-transfer score modes.",
        "required": False,
    },
    {
        "path": "reports/assets/tiled_dinov2_jpeg30_reverse_transfer_detail.csv",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "Per-image JPEG30 tiled DINOv2 scores for robustness diagnostics.",
        "required": False,
    },
    {
        "path": "reports/ms_cocoai_to_ishu_tuned_fusion_jpeg30_tiled_dinov2_2026_06_14.md",
        "type": "result-note",
        "venues": "WIFS,DFF",
        "purpose": "Fixed reverse SCP-Fusion JPEG30 robustness diagnostic after replacing the DINOv2 target branch with tiled target scores.",
        "required": False,
    },
    {
        "path": "reports/assets/ms_cocoai_to_ishu_tuned_fusion_jpeg30_tiled_dinov2_summary.csv",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "Machine-readable fixed-fusion JPEG30 summary for tiled DINOv2 branch replacement.",
        "required": False,
    },
    {
        "path": "reports/assets/ms_cocoai_to_ishu_tuned_fusion_jpeg30_tiled_dinov2_detail.csv",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "Per-seed fixed-fusion JPEG30 detail for tiled DINOv2 branch replacement.",
        "required": False,
    },
    {
        "path": "reports/tiled_dinov2_resize_half_reverse_transfer_2026_06_14.md",
        "type": "result-note",
        "venues": "WIFS,DFF",
        "purpose": "Resize-half robustness probe for tiled DINOv2 reverse-transfer score aggregation.",
        "required": False,
    },
    {
        "path": "reports/assets/tiled_dinov2_resize_half_reverse_transfer_summary.csv",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "Machine-readable summary for resize-half tiled DINOv2 reverse-transfer score modes.",
        "required": False,
    },
    {
        "path": "reports/assets/tiled_dinov2_resize_half_reverse_transfer_seed_metrics.csv",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "Per-seed metrics for resize-half tiled DINOv2 reverse-transfer score modes.",
        "required": False,
    },
    {
        "path": "reports/assets/tiled_dinov2_resize_half_reverse_transfer_detail.csv",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "Per-image resize-half tiled DINOv2 scores for robustness diagnostics.",
        "required": False,
    },
    {
        "path": "reports/ms_cocoai_to_ishu_tuned_fusion_resize_half_tiled_dinov2_2026_06_14.md",
        "type": "result-note",
        "venues": "WIFS,DFF",
        "purpose": "Fixed reverse SCP-Fusion resize-half robustness diagnostic after replacing the DINOv2 target branch with tiled target scores.",
        "required": False,
    },
    {
        "path": "reports/assets/ms_cocoai_to_ishu_tuned_fusion_resize_half_tiled_dinov2_summary.csv",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "Machine-readable fixed-fusion resize-half summary for tiled DINOv2 branch replacement.",
        "required": False,
    },
    {
        "path": "reports/assets/ms_cocoai_to_ishu_tuned_fusion_resize_half_tiled_dinov2_detail.csv",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "Per-seed fixed-fusion resize-half detail for tiled DINOv2 branch replacement.",
        "required": False,
    },
    {
        "path": "reports/tiled_dinov2_screenshot_reverse_transfer_2026_06_14.md",
        "type": "result-note",
        "venues": "WIFS,DFF",
        "purpose": "Screenshot-style robustness probe for tiled DINOv2 reverse-transfer score aggregation.",
        "required": False,
    },
    {
        "path": "reports/assets/tiled_dinov2_screenshot_reverse_transfer_summary.csv",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "Machine-readable summary for screenshot-style tiled DINOv2 reverse-transfer score modes.",
        "required": False,
    },
    {
        "path": "reports/assets/tiled_dinov2_screenshot_reverse_transfer_seed_metrics.csv",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "Per-seed metrics for screenshot-style tiled DINOv2 reverse-transfer score modes.",
        "required": False,
    },
    {
        "path": "reports/assets/tiled_dinov2_screenshot_reverse_transfer_detail.csv",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "Per-image screenshot-style tiled DINOv2 scores for robustness diagnostics.",
        "required": False,
    },
    {
        "path": "reports/ms_cocoai_to_ishu_tuned_fusion_screenshot_tiled_dinov2_2026_06_14.md",
        "type": "result-note",
        "venues": "WIFS,DFF",
        "purpose": "Fixed reverse SCP-Fusion screenshot-style robustness diagnostic after replacing the DINOv2 target branch with tiled target scores.",
        "required": False,
    },
    {
        "path": "reports/assets/ms_cocoai_to_ishu_tuned_fusion_screenshot_tiled_dinov2_summary.csv",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "Machine-readable fixed-fusion screenshot-style summary for tiled DINOv2 branch replacement.",
        "required": False,
    },
    {
        "path": "reports/assets/ms_cocoai_to_ishu_tuned_fusion_screenshot_tiled_dinov2_detail.csv",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "Per-seed fixed-fusion screenshot-style detail for tiled DINOv2 branch replacement.",
        "required": False,
    },
    {
        "path": "reports/tiled_dinov2_transform_stress_comparison_2026_06_14.md",
        "type": "result-note",
        "venues": "WIFS,DFF",
        "purpose": "Core-transform comparison of blur1, JPEG30, resize-half, and screenshot-style tiled-DINO fusion stress probes.",
        "required": False,
    },
    {
        "path": "reports/assets/tiled_dinov2_transform_stress_comparison.csv",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "Machine-readable blur1, JPEG30, resize-half, and screenshot-style tiled-DINO fusion stress deltas.",
        "required": False,
    },
    {
        "path": "reports/tiled_dinov2_calibration_tradeoff_2026_06_14.md",
        "type": "result-note",
        "venues": "WIFS,DFF",
        "purpose": "Calibration tradeoff report separating tiled-DINO decision/ranking gains from Brier/ECE behavior.",
        "required": False,
    },
    {
        "path": "reports/assets/tiled_dinov2_calibration_tradeoff.csv",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "Machine-readable all-mode calibration deltas for tiled-DINO transform probes.",
        "required": False,
    },
    {
        "path": "reports/assets/tiled_dinov2_calibration_tradeoff_choices.csv",
        "type": "table",
        "venues": "WIFS,DFF",
        "purpose": "Machine-readable per-transform mode choices for tiled-DINO accuracy, AUC, Brier, and ECE.",
        "required": False,
    },
]

VENUES = [
    {
        "venue": "DFRWS-USA 2026 poster/demo",
        "deadline": "2026-07-07",
        "status": "ready to polish",
        "title": "When AI Image Detectors Travel: Source-Heldout Diagnostics for Physical, Neural, and Frozen-Encoder Forensics",
        "claim": "Lead with source-heldout diagnostics, CLIP transfer ranking, triage, robustness caveats, and public reproducibility.",
        "next_action": "Finalize poster layout with the selected seed-29 qualitative grid and export the final PDF/poster image.",
    },
    {
        "venue": "IEEE WIFS 2026 paper",
        "deadline": "2026-07-15",
        "status": "feasible with tighter breadth",
        "title": "Source-Heldout Evaluation of Physical, Neural, and Frozen-Encoder Signals for AI-Generated Image Detection",
        "claim": "Use a compact benchmark-paper framing around metric splits: ranking, calibration, fake-call rate, and source-aware decisions.",
        "next_action": "Draft the 6-page paper with the frozen WIFS breadth stance; keep any source-aware v4 work appendix-only.",
    },
    {
        "venue": "DFF-2026 ACM Multimedia workshop",
        "deadline": "2026-07-16",
        "status": "best full-paper fit",
        "title": "SCP-Fusion: Source-Calibrated Physical and Foundation Features for Robust AI-Generated Image Forensics",
        "claim": "Frame SCP-Fusion as a diagnostic fusion protocol for robustness, dataset bias, explainability, and real-world processing.",
        "next_action": "Turn failure grids and source-slice diagnostics into a short explainability/failure-analysis section.",
    },
]

REGEN_COMMANDS = [
    ("literature map", "python scripts/build_literature_map.py"),
    ("draft BibTeX references", "python scripts/build_references_bib.py"),
    ("source-heldout model selection", "python scripts/select_reverse_fusion_by_source_holdout.py"),
    ("source-heldout stress figure", "python scripts/build_source_stress_figure.py"),
    ("publication tables", "python scripts/build_publication_tables.py"),
    ("robustness failure ranking", "python scripts/build_robustness_failure_ranking.py"),
    ("publication figures", "python scripts/build_publication_assets.py"),
    ("opportunity watchlist", "python scripts/build_opportunity_watchlist.py"),
    ("external benchmark readiness", "python scripts/build_external_benchmark_readiness.py"),
    ("external benchmark claim lint", "python scripts/lint_external_benchmark_claims.py"),
    ("SOTA gap report", "python scripts/build_sota_gap_report.py"),
    ("SOTA gap closure plan", "python scripts/build_sota_gap_closure_plan.py"),
    ("competition submission dry run", "python scripts/build_competition_submission_dry_run.py"),
    ("reconstruction-lite feature report", "python scripts/build_reconstruction_lite_feature_report.py"),
    ("reconstruction-v2 feature report", "python scripts/build_reconstruction_v2_feature_report.py"),
    ("reconstruction-v2 bounded probe", "python scripts/summarize_reconstruction_v2_probe.py"),
    ("reconstruction-lite bounded probe", "python scripts/summarize_reconstruction_lite_probe.py"),
    ("reconstruction-lite transfer probe", "python scripts/summarize_reconstruction_lite_transfer_probe.py"),
    ("reconstruction-lite fusion probe", "python scripts/summarize_reconstruction_lite_fusion_probe.py"),
    ("tiled foundation comparison", "python scripts/build_tiled_foundation_comparison.py"),
    ("tiled foundation fusion comparison", "python scripts/build_tiled_foundation_fusion_comparison.py"),
    (
        "tiled DINO resize-half fusion robustness",
        "python scripts/evaluate_reverse_tiled_fusion_robustness.py --variant resize_half --tile-branch dinov2_vits14 --tile-detail reports/assets/tiled_dinov2_resize_half_reverse_transfer_detail.csv --asset-prefix ms_cocoai_to_ishu_tuned_fusion_resize_half_tiled_dinov2 --report-path reports/ms_cocoai_to_ishu_tuned_fusion_resize_half_tiled_dinov2_2026_06_14.md",
    ),
    (
        "tiled DINO blur1 fusion robustness",
        "python scripts/evaluate_reverse_tiled_fusion_robustness.py --variant blur1 --tile-branch dinov2_vits14 --tile-detail reports/assets/tiled_dinov2_blur1_reverse_transfer_detail.csv --asset-prefix ms_cocoai_to_ishu_tuned_fusion_blur1_tiled_dinov2 --report-path reports/ms_cocoai_to_ishu_tuned_fusion_blur1_tiled_dinov2_2026_06_14.md",
    ),
    (
        "tiled DINO JPEG30 fusion robustness",
        "python scripts/evaluate_reverse_tiled_fusion_robustness.py --variant jpeg30 --tile-branch dinov2_vits14 --tile-detail reports/assets/tiled_dinov2_jpeg30_reverse_transfer_detail.csv --asset-prefix ms_cocoai_to_ishu_tuned_fusion_jpeg30_tiled_dinov2 --report-path reports/ms_cocoai_to_ishu_tuned_fusion_jpeg30_tiled_dinov2_2026_06_14.md",
    ),
    (
        "tiled DINO screenshot fusion robustness",
        "python scripts/evaluate_reverse_tiled_fusion_robustness.py --variant screenshot --tile-branch dinov2_vits14 --tile-detail reports/assets/tiled_dinov2_screenshot_reverse_transfer_detail.csv --asset-prefix ms_cocoai_to_ishu_tuned_fusion_screenshot_tiled_dinov2 --report-path reports/ms_cocoai_to_ishu_tuned_fusion_screenshot_tiled_dinov2_2026_06_14.md",
    ),
    (
        "tiled DINO transform stress comparison",
        "python scripts/build_tiled_dinov2_transform_stress_comparison.py",
    ),
    (
        "tiled DINO calibration tradeoff",
        "python scripts/build_tiled_dinov2_calibration_tradeoff.py",
    ),
    ("calibration operating modes", "python scripts/build_calibration_operating_modes.py"),
    ("claim matrix", "python scripts/build_claim_evidence_matrix.py"),
    ("claim matrix lint", "python scripts/lint_claim_evidence_matrix.py"),
    ("method family comparison", "python scripts/build_method_family_comparison.py"),
    ("submission result tables", "python scripts/build_submission_result_tables.py"),
    ("submission result table lint", "python scripts/lint_submission_result_tables.py"),
    ("submission LaTeX tables", "python scripts/build_submission_latex_tables.py"),
    ("submission text drafts", "python scripts/build_submission_text_drafts.py"),
    ("paper section drafts", "python scripts/build_paper_section_drafts.py"),
    ("paper section draft lint", "python scripts/lint_paper_section_drafts.py"),
    ("submission paper skeletons", "python scripts/build_submission_paper_skeletons.py"),
    ("paper skeleton lint", "python scripts/lint_paper_skeletons.py"),
    ("DFRWS qualitative grid selection", "python scripts/build_dfrws_qualitative_grid_selection.py"),
    ("DFRWS poster brief", "python scripts/build_dfrws_poster_brief.py"),
    ("DFRWS poster panels", "python scripts/build_dfrws_poster_figures.py"),
    ("DFRWS poster package lint", "python scripts/lint_dfrws_poster_package.py"),
    ("WIFS breadth decision", "python scripts/build_wifs_breadth_decision.py"),
    ("manuscript assembly map", "python scripts/build_manuscript_assembly_map.py"),
    ("manuscript drafts", "python scripts/build_manuscript_drafts.py"),
    ("manuscript draft lint", "python scripts/lint_manuscript_drafts.py"),
    ("publication control suite dry run", "python scripts/run_publication_control_suite.py --dry-run"),
    ("submission path sanitization", "python scripts/sanitize_submission_local_paths.py --apply"),
    ("submission artifact hashes", "python scripts/build_submission_artifact_hashes.py"),
    ("submission artifact hash lint", "python scripts/lint_submission_artifact_hashes.py"),
    ("submission privacy audit", "python scripts/build_submission_privacy_audit.py"),
    ("submission packet", "python scripts/build_submission_packet.py"),
    ("submission package lint", "python scripts/lint_submission_package.py"),
    ("submission scorecard", "python scripts/build_submission_scorecard.py"),
    ("submission upload checklist", "python scripts/build_submission_upload_checklist.py"),
    ("submission critical path", "python scripts/build_submission_critical_path.py"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a venue-facing submission packet manifest from checked-in reports."
    )
    parser.add_argument("--repo-root", default=".", help="Repository root for validating relative artifact paths.")
    parser.add_argument(
        "--core-results",
        default="reports/assets/publication_core_results.csv",
        help="Core result table generated by build_publication_tables.py.",
    )
    parser.add_argument(
        "--claim-matrix",
        default="reports/assets/claim_evidence_matrix.csv",
        help="Claim/evidence matrix generated by build_claim_evidence_matrix.py.",
    )
    parser.add_argument(
        "--out-path",
        default="reports/submission_packet_2026_06_13.md",
        help="Markdown packet report to write.",
    )
    parser.add_argument(
        "--manifest-out",
        default="reports/assets/submission_packet_manifest.csv",
        help="Machine-readable artifact manifest to write.",
    )
    parser.add_argument(
        "--run-date",
        default=DEFAULT_RUN_DATE.isoformat(),
        help="Date to stamp into the generated packet, in YYYY-MM-DD format. Defaults to today's local date.",
    )
    return parser.parse_args()


def _metric_summary(row: pd.Series) -> str:
    parts = []
    for column, label in [
        ("accuracy", "accuracy"),
        ("auc", "AUC"),
        ("brier", "Brier"),
        ("ece", "ECE"),
        ("predicted_fake_rate", "fake-call rate"),
        ("coverage", "coverage"),
        ("decided_accuracy", "decided accuracy"),
    ]:
        value = row.get(column)
        if pd.notna(value):
            parts.append(f"{label} {float(value):.4f}")
    return " / ".join(parts)


def _rows_by_id(frame: pd.DataFrame, finding_ids: list[str]) -> pd.DataFrame:
    rows = []
    for finding_id in finding_ids:
        matches = frame[frame["finding_id"] == finding_id]
        if matches.empty:
            raise ValueError(f"Missing finding_id={finding_id!r}")
        rows.append(matches.iloc[0])
    return pd.DataFrame(rows)


def _markdown_escape(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).replace("\n", " ").replace("|", "\\|")


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame[columns].itertuples(index=False):
        lines.append("| " + " | ".join(_markdown_escape(value) for value in row) + " |")
    return "\n".join(lines)


def _artifact_manifest(repo_root: Path) -> pd.DataFrame:
    rows = []
    for artifact in ARTIFACTS:
        artifact_path = repo_root / artifact["path"]
        rows.append(
            {
                **artifact,
                "exists": artifact_path.exists(),
                "size_bytes": artifact_path.stat().st_size if artifact_path.exists() else pd.NA,
            }
        )
    manifest = pd.DataFrame(rows)
    missing = manifest[(manifest["required"]) & (~manifest["exists"])]
    if not missing.empty:
        missing_paths = ", ".join(missing["path"].tolist())
        raise FileNotFoundError(f"Missing required submission artifacts: {missing_paths}")
    return manifest


def _lead_result_table(core_results: pd.DataFrame) -> pd.DataFrame:
    rows = _rows_by_id(core_results, LEAD_FINDINGS)
    return pd.DataFrame(
        {
            "finding": rows["finding_id"],
            "method": rows["method"],
            "setting": rows["setting"],
            "metrics": [_metric_summary(row) for _index, row in rows.iterrows()],
            "interpretation": rows["interpretation"],
        }
    )


def _claim_table(claim_matrix: pd.DataFrame) -> pd.DataFrame:
    ready = claim_matrix[claim_matrix["status"].isin(["ready", "ready_with_caveat"])].copy()
    return ready[["claim_id", "status", "submission_use", "risk_or_caveat"]]


def _venue_table(manifest: pd.DataFrame, venue_key: str) -> pd.DataFrame:
    mask = manifest["venues"].str.split(",").apply(lambda values: venue_key in values)
    return manifest[mask][["type", "path", "purpose"]]


def build_submission_packet(
    repo_root: Path,
    core_results: Path,
    claim_matrix: Path,
    run_date: date = DEFAULT_RUN_DATE,
) -> tuple[str, pd.DataFrame]:
    repo_root = repo_root.resolve()
    core = pd.read_csv(core_results)
    claims = pd.read_csv(claim_matrix)
    manifest = _artifact_manifest(repo_root)
    lead = _lead_result_table(core)
    claim_display = _claim_table(claims)
    venue_frame = pd.DataFrame(VENUES)
    commands = pd.DataFrame(REGEN_COMMANDS, columns=["asset", "command"])

    lines = [
        "# Submission Packet Manifest",
        "",
        f"Run date: {run_date.isoformat()}",
        "",
        "Generated by `scripts/build_submission_packet.py` from checked-in reports, figures, and claim tables.",
        "",
        "This packet is an authoring map, not a substitute for venue-specific formatting. It validates that the files currently cited for DFRWS, WIFS, and DFF exist in the public repo.",
        "",
        "## Venue Status",
        "",
        _markdown_table(venue_frame, ["venue", "deadline", "status", "title", "claim", "next_action"]),
        "",
        "## Lead Results To Carry",
        "",
        _markdown_table(lead, ["finding", "method", "setting", "metrics", "interpretation"]),
        "",
        "## Claims And Caveats",
        "",
        _markdown_table(claim_display, list(claim_display.columns)),
        "",
        "## DFRWS Packet",
        "",
        _markdown_table(_venue_table(manifest, "DFRWS"), ["type", "path", "purpose"]),
        "",
        "## WIFS Packet",
        "",
        _markdown_table(_venue_table(manifest, "WIFS"), ["type", "path", "purpose"]),
        "",
        "## DFF Packet",
        "",
        _markdown_table(_venue_table(manifest, "DFF"), ["type", "path", "purpose"]),
        "",
        "## Regeneration Commands",
        "",
        _markdown_table(commands, ["asset", "command"]),
        "",
        "## Current Editorial Guardrails",
        "",
        "- Do not claim classic multi-light photometric stereo; the current physical branch is a single-image proxy.",
        "- Do not claim SCP-Fusion universally beats frozen CLIP; CLIP is still the transfer-ranking frontier in the current evidence.",
        "- Do not claim SOTA or NTIRE/ImageCLEF leaderboard placement; the checked SOTA-gap report separates official results from local proxy evidence.",
        "- Treat native/foundation tiling as bounded diagnostic evidence; clean and core-transform stress gains are small and external benchmark coverage is still incomplete.",
        "- Keep `combined_v4` as an ablation unless source-aware feature selection or stronger regularization changes the transfer gate.",
        "",
    ]
    return "\n".join(lines), manifest


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root)
    text, manifest = build_submission_packet(
        repo_root=repo_root,
        core_results=Path(args.core_results),
        claim_matrix=Path(args.claim_matrix),
        run_date=date.fromisoformat(args.run_date),
    )
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    manifest_path = Path(args.manifest_out)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    print(out_path.resolve())
    print(manifest_path.resolve())


if __name__ == "__main__":
    main()
