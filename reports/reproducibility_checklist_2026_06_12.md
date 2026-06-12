# Public Reproducibility Checklist

Run date: 2026-06-12

Repository: https://github.com/DrStrangel0ve/ai-image-forensics-comparison

This checklist is meant for reviewers, poster visitors, and collaborators who want to understand what is reproducible from the public repo and what requires downloading external datasets.

## What Is Public In The Repo

- Source code for dataset export/audit, conventional feature extraction, neural baselines, physics-guided fusion, frozen encoder baselines, score fusion, calibration diagnostics, source-heldout diagnostics, robustness transforms, and publication figure generation.
- Configuration for known Kaggle, Hugging Face, and external dataset candidates in `configs/datasets.json`.
- Checked-in summary reports under `reports/`.
- Checked-in figure/table assets under `reports/assets/`, including calibration curves, triage summaries, qualitative failure-case grids, and `combined_v4` ablation summaries.
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
| DFRWS-style figures and abstract draft | `reports/publication_assets_2026_06_12.md` |
| Physics-guided ResNet vs vanilla ResNet | `reports/physics_guided_vs_resnet_2026_06_12.md` |
| Ishu repeated split and transfer results | `reports/ishu_ai_vs_real_2026_benchmark.md` |
| Ishu transform robustness | `reports/ishu_physics_guided_robustness_3seed.md` |
| MS COCOAI source-balanced validation | `reports/ms_cocoai_source_balanced_validation.md` |
| SCP-Fusion branch-dropout probe | `reports/score_fusion_branch_dropout_probe_2026_06_12.md` |
| SCP-Fusion source-calibration probe | `reports/score_fusion_source_calibration_probe_2026_06_13.md` |
| Calibration diagnostics | `reports/calibration_diagnostics_2026_06_12.md` |
| Source-heldout calibration with bootstrap CIs | `reports/source_holdout_calibration_2026_06_12.md` |
| Source-heldout triage with bootstrap CIs | `reports/source_holdout_triage_2026_06_12.md` |
| Qualitative failure cases | `reports/qualitative_failure_cases_2026_06_12.md` |
| `combined_v4` feature implementation probe | `reports/combined_v4_probe_2026_06_12.md` |
| `combined_v4` feature-selection sweep with bootstrap CIs | `reports/combined_v4_selectk_probe_2026_06_12.md` |
| Medium `combined_v4` ablation grid with bootstrap CIs | `reports/combined_v4_medium_selectk_probe_2026_06_12.md` |

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

Rebuild publication figures:

```powershell
python scripts\build_publication_assets.py --out-dir reports\assets
```

## Current Result Snapshot

- Physics-guided ResNet-18 + `combined_v3` beats vanilla ResNet-18 on Ishu same-domain three-seed means: 0.8450 accuracy / 0.9177 AUC versus 0.8246 / 0.8927.
- Across 12 Ishu robustness checks, physics-guided fusion wins 10 by accuracy and 10 by AUC.
- Vanilla ResNet-18 still wins MS COCOAI in-domain validation: 0.8160 accuracy / 0.8967 AUC versus physics-guided fusion at 0.7800 / 0.8790.
- Frozen ConvNeXt-Tiny is the strongest current same-domain ranking baseline on Ishu: 0.8947 mean accuracy / 0.9589 mean AUC.
- SCP-Fusion v0 improves Ishu-to-MS-COCOAI mean AUC to 0.7282, but default-threshold accuracy remains weak because target-domain fake scores are under-confident.
- SCP-Fusion branch dropout is implemented with coefficient export, but the first probe is negative/mixed: default accuracy is essentially flat while AUC/Brier/ECE worsen.
- Source-calibrated SCP-Fusion improves default accuracy to 0.6073 and Brier/ECE to 0.3123 / 0.2947, with a small AUC tradeoff.
- Source-heldout calibration and triage assets include 95% deterministic bootstrap confidence intervals over held-out source/seed rows.
- Medium `combined_v4` ablation suggests raw `combined_v4` is the best ranking/accuracy candidate at 240 train images per seed, while select-k60 is the best calibrated variant; regenerated ablation assets include 95% bootstrap confidence intervals.

## Known Limitations

- The photometric branch is a single-image proxy, not calibrated multi-light photometric stereo.
- Most reported experiments are practical repeated-seed probes, not exhaustive full-dataset training sweeps.
- External datasets must be downloaded under their own licenses and may change or disappear.
- Some reported source-heldout diagnostics use saved prediction scores rather than full leave-one-generator-out retraining.
- Calibration and threshold transfer remain open problems; strong AUC does not guarantee good default-threshold behavior.

## Public Sharing Notes

The repository is safe to share as code and reports. It should be described as a research scaffold and benchmark package, not as a production detector. The strongest current claim is methodological:

> Robust AI-image forensics needs source-aware evaluation across physical/signal features, learned image representations, frozen foundation encoders, and explicit calibration diagnostics.
