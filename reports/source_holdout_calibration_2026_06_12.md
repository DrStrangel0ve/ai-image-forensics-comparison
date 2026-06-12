# Source-Heldout Post-Hoc Calibration

Run date: 2026-06-12

This follow-up tests whether post-hoc calibration can fix the leave-one-generator-out failure exposed in `source_holdout_diagnostics_2026_06_12.md`. It uses existing Ishu-to-MS-COCOAI prediction scores, holds out one generated MS COCOAI source at a time, fits a calibrator on the remaining generated sources plus the real-image calibration split, and evaluates on the held-out generated source plus the real-image test split.

Implementation:

- `scripts/summarize_source_holdout_calibration.py`
- calibrators:
  - temperature scaling;
  - class-balanced temperature scaling;
  - Platt/logit calibration;
  - class-balanced Platt/logit calibration;
  - isotonic regression;
  - class-balanced isotonic regression.
- seeds: 7, 17, 29
- held-out generated sources per seed: SD2.1, SDXL, SD3, DALL-E 3, Midjourney 6
- summary CSVs now include 95% deterministic bootstrap confidence intervals over the 15 held-out source/seed rows.

## Main Result

Balanced temperature scaling is the safest calibration pass. It improves Brier score and ECE for the strongest neural/fusion models while preserving the raw `0.5` decision boundary. That means it improves probability quality, but it does not solve the fake-recall problem by itself.

The checked-in summary asset includes confidence bounds for the means. For the strongest calibrated row, SCP-Fusion v0 with class-balanced temperature scaling has calibrated Brier `0.1995` with a 95% bootstrap interval of `[0.1964, 0.2023]`, and calibrated ECE `0.1331` with interval `[0.1034, 0.1638]`.

| method | raw accuracy | calibrated accuracy | raw Brier | calibrated Brier | raw ECE | calibrated ECE | calibrated real FPR | calibrated fake detection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SCP-Fusion v0 | 0.7417 | 0.7417 | 0.2009 | 0.1995 | 0.1532 | 0.1331 | 0.0533 | 0.2293 |
| frozen ConvNeXt-Tiny | 0.7429 | 0.7429 | 0.2157 | 0.2002 | 0.1943 | 0.1441 | 0.0813 | 0.3033 |
| physics-guided fusion | 0.6943 | 0.6943 | 0.2557 | 0.2158 | 0.2347 | 0.1513 | 0.1787 | 0.3767 |
| ResNet-18 | 0.6787 | 0.6787 | 0.2671 | 0.2172 | 0.2447 | 0.1483 | 0.1800 | 0.3253 |
| `combined_v3` | 0.6811 | 0.6811 | 0.2382 | 0.2368 | 0.1785 | 0.1861 | 0.1347 | 0.2207 |

## Negative Result

More flexible calibrators are risky under source shift. Unbalanced Platt and isotonic calibration often learn the non-heldout source prior too aggressively, push many scores above `0.5`, and create high real-image false-positive rates:

| method | calibrator | calibrated accuracy | calibrated Brier | calibrated ECE | calibrated real FPR | calibrated fake detection |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| SCP-Fusion v0 | Platt | 0.5985 | 0.2648 | 0.2896 | 0.4579 | 0.7393 |
| frozen ConvNeXt-Tiny | Platt | 0.4975 | 0.2722 | 0.2911 | 0.6240 | 0.8013 |
| physics-guided fusion | Platt | 0.4497 | 0.2944 | 0.3115 | 0.7029 | 0.8313 |
| ResNet-18 | Platt | 0.4278 | 0.2968 | 0.3146 | 0.7443 | 0.8580 |
| `combined_v3` | Platt | 0.3021 | 0.3108 | 0.3252 | 0.9448 | 0.9193 |

Balanced Platt and isotonic variants are less extreme, but they still usually trade away too much real-image specificity. That makes them useful diagnostics, not good default deployment choices.

## Interpretation

This splits the calibration story into two parts:

- Balanced temperature scaling is a strong paper-ready post-hoc calibration baseline. It improves Brier/ECE without changing ranking or default decisions.
- It does not fix source-heldout fake recall. SCP-Fusion v0 still detects only 22.9% of held-out fake images at threshold `0.5`, despite having the best Brier score.
- Flexible calibrators can look attractive on non-heldout sources but overfit the source/prior mixture and inflate real false positives.

For SCP-Fusion v1, the next target is a calibration-aware training objective or validation protocol, not just a post-hoc calibrator. Good candidates:

- source-heldout validation loss;
- source-balanced calibration batches;
- a temperature head selected by leave-one-generator validation;
- a two-threshold triage mode that separates "likely fake", "uncertain", and "likely real".

## Reproduce

```powershell
python scripts/summarize_source_holdout_calibration.py `
  --metadata data\raw\ms_cocoai_2026_validation_source_balanced_100\metadata.csv `
  --out-dir runs\source_holdout_calibration\ishu_to_ms_cocoai_all4 `
  --split validation `
  --seed 7 `
  --n-bins 10 `
  --calibrator temperature `
  --calibrator temperature_balanced `
  --calibrator platt `
  --calibrator platt_balanced `
  --calibrator isotonic `
  --calibrator isotonic_balanced `
  --predictions seed7:combined_v3=runs\ishu_to_ms_cocoai_source_balanced_seed7\combined_v3\predictions.csv `
  --predictions seed7:resnet18=runs\ishu_to_ms_cocoai_source_balanced_seed7\resnet18\predictions.csv `
  --predictions seed7:physics_guided=runs\ishu_to_ms_cocoai_source_balanced_seed7\physics_guided_resnet18_combined_v3\predictions.csv `
  --predictions seed7:convnext_tiny_frozen=runs\ishu_to_ms_cocoai_source_balanced_seed7\convnext_tiny_frozen\predictions.csv `
  --predictions seed7:scp_fusion_v0=runs\score_fusion\ishu_seed7_to_ms_cocoai_all4\ms_cocoai\predictions.csv `
  --predictions seed17:combined_v3=runs\ishu_to_ms_cocoai_source_balanced_seed17\combined_v3\predictions.csv `
  --predictions seed17:resnet18=runs\ishu_to_ms_cocoai_source_balanced_seed17\resnet18\predictions.csv `
  --predictions seed17:physics_guided=runs\ishu_to_ms_cocoai_source_balanced_seed17\physics_guided_resnet18_combined_v3\predictions.csv `
  --predictions seed17:convnext_tiny_frozen=runs\ishu_to_ms_cocoai_source_balanced_seed17\convnext_tiny_frozen\predictions.csv `
  --predictions seed17:scp_fusion_v0=runs\score_fusion\ishu_seed17_to_ms_cocoai_all4\ms_cocoai\predictions.csv `
  --predictions seed29:combined_v3=runs\ishu_to_ms_cocoai_source_balanced_seed29\combined_v3\predictions.csv `
  --predictions seed29:resnet18=runs\ishu_to_ms_cocoai_source_balanced_seed29\resnet18\predictions.csv `
  --predictions seed29:physics_guided=runs\ishu_to_ms_cocoai_source_balanced_seed29\physics_guided_resnet18_combined_v3\predictions.csv `
  --predictions seed29:convnext_tiny_frozen=runs\ishu_to_ms_cocoai_source_balanced_seed29\convnext_tiny_frozen\predictions.csv `
  --predictions seed29:scp_fusion_v0=runs\score_fusion\ishu_seed29_to_ms_cocoai_all4\ms_cocoai\predictions.csv
```

The run writes the full summary CSV to `runs\source_holdout_calibration\ishu_to_ms_cocoai_all4\source_holdout_calibration_summary.csv`. A copy is checked in at `reports\assets\source_holdout_calibration_summary_ishu_ms_cocoai_all4.csv`.
