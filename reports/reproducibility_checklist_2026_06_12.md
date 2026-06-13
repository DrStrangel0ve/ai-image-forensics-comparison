# Public Reproducibility Checklist

Run date: 2026-06-12

Repository: https://github.com/DrStrangel0ve/ai-image-forensics-comparison

This checklist is meant for reviewers, poster visitors, and collaborators who want to understand what is reproducible from the public repo and what requires downloading external datasets.

## What Is Public In The Repo

- Source code for dataset export/audit, conventional feature extraction, neural baselines, physics-guided fusion, frozen encoder baselines, score fusion, calibration diagnostics, source-heldout diagnostics, robustness transforms, and publication figure generation.
- Configuration for known Kaggle, Hugging Face, and external dataset candidates in `configs/datasets.json`.
- Checked-in summary reports under `reports/`.
- Checked-in figure/table assets under `reports/assets/`, including calibration curves, triage summaries, qualitative failure-case grids, and `combined_v4` ablation summaries.
- The transform robustness figure is checked in at `reports/assets/publication_reverse_transform_robustness.png`.
- A permissive MIT license and `CITATION.cff`.

## What Is Not Public In The Repo

- Raw datasets. The `data/` directory is intentionally git-ignored because the source datasets have their own licenses, access rules, and sizes.
- Model checkpoints and run directories. The `runs/`, `models/`, `*.pt`, and `*.joblib` artifacts are intentionally git-ignored because they are derived outputs and can be large.
- Kaggle credentials. `kaggle.json` is ignored and should remain in the user's profile, not the repository.

## Main Evidence Artifacts

| claim area | report |
| --- | --- |
| Research roadmap and contest/venue fit | `reports/research_deep_dive_2026_06_12.md` |
| Submission plan for DFRWS, WIFS, and DFF | `reports/submission_plan_2026.md` |
| Submission readiness snapshot | `reports/submission_readiness_2026_06_13.md` |
| Claim evidence matrix for poster/paper claims | `reports/assets/claim_evidence_matrix.md` |
| Publication core results table | `reports/assets/publication_core_results.md` |
| DFRWS-style figures and abstract draft | `reports/publication_assets_2026_06_12.md` |
| Generated DFRWS poster brief | `reports/dfrws_poster_brief_2026_06_13.md` |
| DFRWS poster key numbers CSV | `reports/assets/dfrws_poster_key_numbers.csv` |
| Editable DFRWS poster draft | `reports/dfrws_poster_draft_2026_06_13.md` |
| DFRWS poster draft PPTX | `reports/assets/dfrws_poster_draft_2026_06_13.pptx` |
| DFRWS poster draft PNG preview | `reports/assets/dfrws_poster_draft_2026_06_13.png` |
| DFRWS poster-native transfer and robustness panels | `reports/dfrws_poster_native_figures_2026_06_13.md` |
| Updated DFRWS poster draft v2 | `reports/dfrws_poster_draft_v2_2026_06_13.md` |
| DFRWS poster draft v2 PPTX | `reports/assets/dfrws_poster_draft_v2_2026_06_13.pptx` |
| DFRWS poster draft v2 PNG preview | `reports/assets/dfrws_poster_draft_v2_2026_06_13.png` |
| Physics-guided ResNet vs vanilla ResNet | `reports/physics_guided_vs_resnet_2026_06_12.md` |
| Ishu repeated split and transfer results | `reports/ishu_ai_vs_real_2026_benchmark.md` |
| Ishu transform robustness | `reports/ishu_physics_guided_robustness_3seed.md` |
| MS COCOAI source-balanced validation | `reports/ms_cocoai_source_balanced_validation.md` |
| CLIP/DINO frozen encoder support | `reports/clip_dino_encoder_support_2026_06_13.md` |
| DINOv2 frozen encoder smoke probe | `reports/dinov2_frozen_encoder_smoke_2026_06_13.md` |
| DINOv2 three-seed frozen encoder baseline | `reports/dinov2_three_seed_foundation_baseline_2026_06_13.md` |
| SCP-Fusion + DINOv2 probe | `reports/score_fusion_dinov2_probe_2026_06_13.md` |
| CLIP ViT-B/32 foundation and fusion probe | `reports/clip_vit_b32_foundation_and_fusion_2026_06_13.md` |
| MS COCOAI to Ishu reverse foundation transfer | `reports/ms_cocoai_to_ishu_foundation_reverse_2026_06_13.md` |
| MS COCOAI to Ishu reverse neural and fusion baselines | `reports/ms_cocoai_to_ishu_reverse_neural_fusion_2026_06_13.md` |
| MS COCOAI to Ishu reverse fusion regularization | `reports/ms_cocoai_to_ishu_reverse_fusion_regularization_2026_06_13.md` |
| MS COCOAI to Ishu source-threshold fusion | `reports/ms_cocoai_to_ishu_source_threshold_fusion_2026_06_13.md` |
| MS COCOAI to Ishu threshold tie-break probe | `reports/ms_cocoai_to_ishu_threshold_tiebreak_probe_2026_06_13.md` |
| MS COCOAI to Ishu threshold cap probe | `reports/ms_cocoai_to_ishu_threshold_cap_probe_2026_06_13.md` |
| Source-utility score-fusion threshold strategy | `reports/score_fusion_source_utility_threshold_2026_06_13.md` |
| MS COCOAI to Ishu source-utility threshold sweep | `reports/ms_cocoai_to_ishu_source_utility_threshold_2026_06_13.md` |
| MS COCOAI to Ishu source-utility model selection | `reports/ms_cocoai_to_ishu_model_utility_selection_2026_06_13.md` |
| MS COCOAI to Ishu source-heldout model selection | `reports/ms_cocoai_to_ishu_source_holdout_model_selection_2026_06_13.md` |
| MS COCOAI to Ishu source-heldout tuned fusion | `reports/ms_cocoai_to_ishu_source_holdout_tuned_fusion_2026_06_13.md` |
| MS COCOAI to Ishu tuned-fusion constraint sweep | `reports/ms_cocoai_to_ishu_tuned_fusion_constraint_sweep_2026_06_13.md` |
| MS COCOAI to Ishu tuned-fusion JPEG70 robustness | `reports/ms_cocoai_to_ishu_tuned_fusion_jpeg70_robustness_2026_06_13.md` |
| MS COCOAI to Ishu tuned-fusion JPEG50 robustness | `reports/ms_cocoai_to_ishu_tuned_fusion_jpeg50_robustness_2026_06_13.md` |
| MS COCOAI to Ishu tuned-fusion JPEG30 robustness | `reports/ms_cocoai_to_ishu_tuned_fusion_jpeg30_robustness_2026_06_13.md` |
| MS COCOAI to Ishu tuned-fusion blur robustness | `reports/ms_cocoai_to_ishu_tuned_fusion_blur1_robustness_2026_06_13.md` |
| MS COCOAI to Ishu tuned-fusion resize robustness | `reports/ms_cocoai_to_ishu_tuned_fusion_resize_half_robustness_2026_06_13.md` |
| MS COCOAI to Ishu tuned-fusion crop robustness | `reports/ms_cocoai_to_ishu_tuned_fusion_crop85_robustness_2026_06_13.md` |
| MS COCOAI to Ishu tuned-fusion noise robustness | `reports/ms_cocoai_to_ishu_tuned_fusion_noise3_robustness_2026_06_13.md` |
| MS COCOAI to Ishu tuned-fusion screenshot robustness | `reports/ms_cocoai_to_ishu_tuned_fusion_screenshot_robustness_2026_06_13.md` |
| MS COCOAI to Ishu tuned-fusion social-square robustness | `reports/ms_cocoai_to_ishu_tuned_fusion_social_square_robustness_2026_06_13.md` |
| Ishu to MS threshold objective sensitivity | `reports/ishu_to_ms_threshold_objective_sweep_2026_06_13.md` |
| SCP-Fusion branch-dropout probe | `reports/score_fusion_branch_dropout_probe_2026_06_12.md` |
| SCP-Fusion source-calibration probe | `reports/score_fusion_source_calibration_probe_2026_06_13.md` |
| SCP-Fusion source-heldout calibration probe | `reports/score_fusion_source_holdout_probe_2026_06_13.md` |
| SCP-Fusion source-heldout utility-tuned triage probe | `reports/score_fusion_triage_tuning_2026_06_13.md` |
| Calibration diagnostics | `reports/calibration_diagnostics_2026_06_12.md` |
| Source-heldout calibration with bootstrap CIs | `reports/source_holdout_calibration_2026_06_12.md` |
| Source-heldout triage with bootstrap CIs | `reports/source_holdout_triage_2026_06_12.md` |
| Qualitative failure cases | `reports/qualitative_failure_cases_2026_06_12.md` |
| `combined_v4` feature implementation probe | `reports/combined_v4_probe_2026_06_12.md` |
| `combined_v4` feature-selection sweep with bootstrap CIs | `reports/combined_v4_selectk_probe_2026_06_12.md` |
| Medium `combined_v4` ablation grid with bootstrap CIs | `reports/combined_v4_medium_selectk_probe_2026_06_12.md` |
| `combined_v4` transfer-readiness gate and command manifest | `reports/combined_v4_transfer_readiness_2026_06_13.md` |
| `combined_v4` full-transfer seed summary | `reports/combined_v4_full_transfer_summary_2026_06_13.md` |
| `combined_v4` source-slice diagnostics | `reports/combined_v4_source_slice_diagnostics_2026_06_13.md` |

## Core Reproduction Commands

Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Optional CUDA wheels for an RTX 3060 Ti style desktop:

```powershell
python -m pip install --force-reinstall -r requirements-gpu-cu128.txt
```

Run tests:

```powershell
.\.venv\Scripts\python -m pytest
```

List dataset candidates:

```powershell
python scripts\list_datasets.py
```

Download the Ishu benchmark dataset:

```powershell
python scripts\download_dataset.py ishu_ai_vs_real_2026
```

Run a medium `combined_v4` ablation grid:

```powershell
python scripts\run_feature_ablation_grid.py `
  --data-dir data\raw\ishu_ai_vs_real_2026 `
  --out-dir runs\combined_v4_medium_selectk_probe `
  --summary-dir reports\assets\combined_v4_medium_selectk_probe `
  --seeds 7 17 29 `
  --image-size 64 `
  --max-train-samples 240 `
  --max-test-samples 120 `
  --skip-errors `
  --skip-existing
```

Rebuild publication figures, core tables, and claim audit:

```powershell
python scripts\build_publication_assets.py --out-dir reports\assets
python scripts\build_publication_tables.py --out-dir reports\assets
python scripts\build_claim_evidence_matrix.py --out-dir reports\assets
python scripts\build_dfrws_poster_brief.py
python scripts\build_dfrws_poster_figures.py
python scripts\build_combined_v4_transfer_readiness.py
python scripts\summarize_combined_v4_transfer.py
python scripts\analyze_combined_v4_source_slices.py
```

## Current Result Snapshot

- Physics-guided ResNet-18 + `combined_v3` beats vanilla ResNet-18 on Ishu same-domain three-seed means: 0.8450 accuracy / 0.9177 AUC versus 0.8246 / 0.8927.
- Across 12 Ishu robustness checks, physics-guided fusion wins 10 by accuracy and 10 by AUC.
- Vanilla ResNet-18 still wins MS COCOAI in-domain validation: 0.8160 accuracy / 0.8967 AUC versus physics-guided fusion at 0.7800 / 0.8790.
- Frozen ConvNeXt-Tiny was the first strong frozen-encoder baseline on Ishu: 0.8947 mean accuracy / 0.9589 mean AUC.
- SCP-Fusion v0 improves Ishu-to-MS-COCOAI mean AUC to 0.7282, and adding frozen DINOv2-small as a fifth branch raises mean AUC to 0.7503.
- Frozen CLIP ViT-B/32 is now the strongest standalone cross-domain ranker: 0.6363 accuracy / 0.8641 AUC on Ishu-to-MS-COCOAI, with source-heldout triage reaching about 47% coverage at about 92.6% decided-case accuracy under the strict 5% budget.
- In the reverse MS-COCOAI-to-Ishu foundation check, CLIP still has the best mean AUC at 0.8243, while ConvNeXt has the best default/source-threshold accuracy at 0.6579 / 0.6784.
- In the reverse all-method check, all-branch fusion has the best mean AUC at 0.8285, while physics-guided ResNet-18 has the best default/source-threshold accuracy at 0.6871 / 0.6813 and the best Brier/ECE.
- Reverse fusion regularization raises the AUC frontier to 0.8406 with branch dropout and improves fusion Brier/ECE to 0.2213 / 0.2091 with strong regularization.
- Capped held-out source-threshold fusion with strong regularization is now the best reverse fusion operating point: 0.7222 accuracy / 0.8291 AUC / 0.2188 Brier on MS-COCOAI-to-Ishu.
- Source-utility model selection is a useful negative result: unconstrained source utility selects over-firing fusion heads at 0.6520 accuracy / 0.8216 target fake-call rate, while a 0.48 source fake-rate cap recovers 0.7193 accuracy but does not beat the fixed capped threshold family.
- Leave-one-generator-out source utility reaches the same conclusion: source-heldout utility alone still selects over-firing fusion heads, while the 0.48 source fake-rate cap recovers 0.7193 accuracy / 0.8291 AUC.
- Source-heldout tuned fusion is the first training-side constrained utility win: 0.7339 accuracy / 0.8341 AUC / 0.2748 Brier on MS-COCOAI-to-Ishu, with remaining target fake-call bias at 0.6813.
- Sweeping tuned-fusion source fake-rate caps gives the best reverse operating point so far: cap 0.40 reaches 0.7632 accuracy / 0.8361 AUC and cuts the target fake-call rate to 0.5175.
- The tuned-fusion cap_0p4 operating point is stable under JPEG70 target recompression at 0.7661 accuracy / 0.8485 AUC, JPEG50 at 0.7515 / 0.8309, noise3 at 0.7690 / 0.8704, and social_square at 0.7778 / 0.8474; it is weaker under JPEG30 at 0.7076 / 0.8167, blur1 at 0.7105 / 0.7872, resize_half at 0.7164 / 0.7816, crop85 at 0.7251 / 0.8227, and screenshot-style roundtrip at 0.7310 / 0.7965.
- The opposite-direction Ishu-to-MS threshold objective sweep improves all-foundation SCP-Fusion accuracy from 0.6163 to 0.6470 on saved scores, but still under-calls generated images with a 0.1863 target fake-call rate.
- All-foundation SCP-Fusion with CLIP and DINOv2 reaches 0.7995 transfer AUC, improving the fusion family while still trailing standalone CLIP.
- SCP-Fusion branch dropout is implemented with coefficient export, but the first probe is negative/mixed: default accuracy is essentially flat while AUC/Brier/ECE worsen.
- Source-calibrated SCP-Fusion improves default accuracy to 0.6073 and Brier/ECE to 0.3123 / 0.2947, with a small AUC tradeoff.
- Source-calibrated DINOv2 fusion is now the best current fusion operating-point result: 0.6127 target accuracy / 0.3062 Brier / 0.2938 ECE.
- Under source-heldout triage, DINOv2-enhanced SCP-Fusion improves the strict 5% operating point to about 26% coverage with about 80% decided-case accuracy.
- Source-heldout calibration and triage assets include 95% deterministic bootstrap confidence intervals over held-out source/seed rows.
- Medium `combined_v4` ablation suggests raw `combined_v4` is the best ranking/accuracy candidate at 240 train images per seed, while select-k60 is the best calibrated variant; regenerated ablation assets include 95% bootstrap confidence intervals.
- The completed `combined_v4` transfer gate keeps `combined_v3` as the main conventional baseline. Raw v4 improves Ishu-to-MS-COCOAI transfer accuracy by +0.0133 but leaves AUC flat and worsens Brier/ECE; select-k60 improves transfer AUC by +0.0119 and Brier/ECE by -0.0156 / -0.0249, but loses -0.0292 same-domain Ishu accuracy.
- Source-slice diagnostics explain the v4 caveat: select-k60 improves transfer detection most on MidJourney v6 (+0.0500) and DALL-E 3 (+0.0400), while same-domain Ishu accuracy drops most on food (-0.0833), items (-0.0571), and animals (-0.0521).

## Known Limitations

- The photometric branch is a single-image proxy, not calibrated multi-light photometric stereo.
- Most reported experiments are practical repeated-seed probes, not exhaustive full-dataset training sweeps.
- External datasets must be downloaded under their own licenses and may change or disappear.
- Some reported source-heldout diagnostics use saved prediction scores rather than full leave-one-generator-out retraining.
- Calibration and threshold transfer remain open problems; strong AUC does not guarantee good default-threshold behavior.

## Public Sharing Notes

The repository is safe to share as code and reports. It should be described as a research scaffold and benchmark package, not as a production detector. The strongest current claim is methodological:

> Robust AI-image forensics needs source-aware evaluation across physical/signal features, learned image representations, frozen foundation encoders, and explicit calibration diagnostics.
