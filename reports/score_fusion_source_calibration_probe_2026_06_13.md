# SCP-Fusion Source-Calibration Probe

Run date: 2026-06-13

This follow-up turns the calibration lesson from the branch-dropout probe into a reusable score-fusion path. Instead of only regularizing the logistic stack, `scripts/fuse_prediction_scores.py` can now reserve a deterministic class-balanced source split, fit the score-fusion head on the remaining source rows, and fit a post-hoc calibrator on held-out source fused scores.

Implementation:

- new shared module: `src/forensic_compare/calibration.py`
- reused by:
  - `scripts/fuse_prediction_scores.py`
  - `scripts/summarize_source_holdout_calibration.py`
  - `scripts/summarize_source_holdout_triage.py`
- new score-fusion options:
  - `--score-calibrator`
  - `--calibration-fraction`
- calibrated fusion outputs now include:
  - calibrated `fake_score`;
  - `raw_fake_score`;
  - `score_calibrator`;
  - Brier/ECE columns in `summary.csv`;
  - saved `score_calibrator.joblib`.

## Three-Seed Result

Configuration:

- source dataset: Ishu AI-vs-real 2026;
- target dataset: source-balanced MS COCOAI / Defactify validation;
- seeds: 7, 17, 29;
- base branches: `combined_v3`, ResNet-18, physics-guided ResNet-18 + `combined_v3`, frozen ConvNeXt-Tiny;
- calibrator: class-balanced temperature scaling;
- calibration fraction: `0.5` of source validation rows, stratified by class.

| method | mean accuracy | mean AUC | mean Brier | mean ECE | predicted fake rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| source-calibrated fusion | 0.6073 | 0.7242 | 0.3123 | 0.2947 | 0.1747 |
| SCP-Fusion v0 | 0.5910 | 0.7282 | 0.3190 | 0.3087 | 0.1383 |
| branch-dropout fusion | 0.5923 | 0.7250 | 0.3389 | 0.3325 | 0.1403 |

The calibrated score-fusion path improves default-threshold accuracy by `+0.0163` over SCP-Fusion v0 and also improves Brier score and ECE. It gives up `0.0039` AUC, so it should be framed as a calibration/operating-point improvement rather than a ranking improvement.

## Fitted Temperatures

| seed | calibrator | calibration fraction | fusion train rows | calibration rows | temperature |
| ---: | --- | ---: | ---: | ---: | ---: |
| 7 | temperature_balanced | 0.5 | 57 | 57 | 0.7619 |
| 17 | temperature_balanced | 0.5 | 57 | 57 | 0.9548 |
| 29 | temperature_balanced | 0.5 | 57 | 57 | 0.7247 |

The source-calibrated model still under-calls generated MS COCOAI images, but less severely than SCP-Fusion v0: predicted fake rate rises from `0.1383` to `0.1747` on a balanced target split.

## Interpretation

This is a stronger SCP-Fusion v1 candidate than score-level branch dropout:

- branch dropout mostly changed regularization and did not improve probability quality;
- source calibration directly targets the under-confident target-domain fake scores;
- the current calibration split is source-domain only, so the next step should be source-heldout calibration rather than claiming deployment readiness;
- AUC remains strongest for SCP-Fusion v0, so paper tables should report ranking and calibrated operating-point metrics separately.

For WIFS/DFF, this supports the core thesis: the main challenge is not just finding more branch signals, but selecting operating points that survive source shift.

## Artifacts

Checked-in compact assets:

- `reports/assets/score_fusion_source_calibrated_summary.csv`
- `reports/assets/score_fusion_source_calibrated_metrics.csv`
- `reports/assets/score_fusion_source_calibrated_temperatures.csv`

Local run folders are ignored by Git and can be regenerated under `runs/score_fusion_source_calibrated/`.

## Reproduce

Seed 7 example:

```powershell
python scripts/fuse_prediction_scores.py `
  --out-dir runs\score_fusion_source_calibrated\ishu_seed7_to_ms_cocoai_all4 `
  --seed 7 `
  --score-calibrator temperature_balanced `
  --calibration-fraction 0.5 `
  --train combined_v3=runs\ishu_ai_vs_real_2026_initial\feature_combined_v3_logistic_regression\predictions.csv `
  --train resnet18=runs\ishu_ai_vs_real_2026_initial\resnet18\predictions.csv `
  --train physics_guided=runs\ishu_ai_vs_real_2026_physics_guided_seed7\physics_guided_resnet18_combined_v3\predictions.csv `
  --train convnext_tiny_frozen=runs\ishu_ai_vs_real_2026_frozen_encoder\convnext_tiny_seed7\predictions.csv `
  --variant ms_cocoai:combined_v3=runs\ishu_to_ms_cocoai_source_balanced_seed7\combined_v3\predictions.csv `
  --variant ms_cocoai:resnet18=runs\ishu_to_ms_cocoai_source_balanced_seed7\resnet18\predictions.csv `
  --variant ms_cocoai:physics_guided=runs\ishu_to_ms_cocoai_source_balanced_seed7\physics_guided_resnet18_combined_v3\predictions.csv `
  --variant ms_cocoai:convnext_tiny_frozen=runs\ishu_to_ms_cocoai_source_balanced_seed7\convnext_tiny_frozen\predictions.csv
```

Compare v0, branch dropout, and source-calibrated fusion:

```powershell
python scripts/summarize_calibration_metrics.py `
  --out-dir runs\calibration_diagnostics\score_fusion_source_calibrated_probe `
  --n-bins 10 `
  --predictions seed7:scp_fusion_v0=runs\score_fusion\ishu_seed7_to_ms_cocoai_all4\ms_cocoai\predictions.csv `
  --predictions seed17:scp_fusion_v0=runs\score_fusion\ishu_seed17_to_ms_cocoai_all4\ms_cocoai\predictions.csv `
  --predictions seed29:scp_fusion_v0=runs\score_fusion\ishu_seed29_to_ms_cocoai_all4\ms_cocoai\predictions.csv `
  --predictions seed7:branch_dropout=runs\score_fusion_branch_dropout\ishu_seed7_to_ms_cocoai_all4\ms_cocoai\predictions.csv `
  --predictions seed17:branch_dropout=runs\score_fusion_branch_dropout\ishu_seed17_to_ms_cocoai_all4\ms_cocoai\predictions.csv `
  --predictions seed29:branch_dropout=runs\score_fusion_branch_dropout\ishu_seed29_to_ms_cocoai_all4\ms_cocoai\predictions.csv `
  --predictions seed7:source_calibrated=runs\score_fusion_source_calibrated\ishu_seed7_to_ms_cocoai_all4\ms_cocoai\predictions.csv `
  --predictions seed17:source_calibrated=runs\score_fusion_source_calibrated\ishu_seed17_to_ms_cocoai_all4\ms_cocoai\predictions.csv `
  --predictions seed29:source_calibrated=runs\score_fusion_source_calibrated\ishu_seed29_to_ms_cocoai_all4\ms_cocoai\predictions.csv
```
