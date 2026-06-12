# Reverse Fusion Regularization Probe

Run date: 2026-06-13

This probe follows the MS COCOAI -> Ishu reverse all-method result. The previous all-branch fusion had the best reverse AUC but still behaved like a CLIP-led fake-overcalling detector. This run asks whether simple fusion regularization or branch dropout can move the AUC/calibration tradeoff before adding more model complexity.

## Setup

- source dataset: `data/raw/ms_cocoai_2026_validation_source_balanced_100/validation`
- target dataset: `data/raw/ishu_ai_vs_real_2026`
- target split: deterministic Ishu `test`, matched to seeds `7`, `17`, and `29`
- score branches: `combined_v3`, ResNet-18, physics-guided ResNet-18, ConvNeXt-Tiny, CLIP ViT-B/32, DINOv2-small
- baseline fusion: class-balanced logistic regression on aligned source validation predictions
- tested knobs:
  - lower fusion `C`: `0.1` and `0.03`
  - branch dropout: rate `0.35`, `8` repeats, mean-score fill
  - optional source-heldout temperature-balanced score calibration

## Reverse Target Results

Mean over three seeds on MS COCOAI -> Ishu:

| fusion config | accuracy | AUC | Brier | ECE | predicted fake rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| branch dropout, mean fill | 0.6520 | 0.8406 | 0.3057 | 0.3275 | 0.8158 |
| branch dropout + temperature | 0.6520 | 0.8403 | 0.2939 | 0.3229 | 0.8099 |
| `C=0.1` | 0.6491 | 0.8366 | 0.2472 | 0.2653 | 0.8129 |
| `C=0.1` + temperature | 0.6579 | 0.8323 | 0.2794 | 0.2941 | 0.7924 |
| `C=0.03` | 0.6608 | 0.8315 | 0.2213 | 0.2091 | 0.7895 |
| baseline + temperature | 0.6579 | 0.8285 | 0.3067 | 0.3304 | 0.8099 |
| baseline | 0.6520 | 0.8275 | 0.2944 | 0.3181 | 0.8216 |

Branch dropout gives the new reverse AUC frontier: `0.8406`, up from `0.8285` for the previous all-branch temperature run and `0.8243` for standalone CLIP. Strong regularization with `C=0.03` is the best probability-quality fusion setting, cutting Brier/ECE to `0.2213 / 0.2091`.

For context, the previous physics-guided ResNet-18 reverse result remains the best default-threshold detector: `0.6871` accuracy and `0.1902` ECE. But the `C=0.03` fusion now beats physics-guided Brier score (`0.2213` versus `0.2436`) while keeping much higher AUC (`0.8315` versus `0.7420`).

## Source-Threshold Calibration

| fusion config | default accuracy | source-threshold accuracy | oracle accuracy | AUC |
| --- | ---: | ---: | ---: | ---: |
| branch dropout, mean fill | 0.6520 | 0.6579 | 0.7807 | 0.8406 |
| branch dropout + temperature | 0.6520 | 0.6433 | 0.7778 | 0.8403 |
| `C=0.1` | 0.6491 | 0.6754 | 0.7807 | 0.8366 |
| `C=0.1` + temperature | 0.6579 | 0.6842 | 0.7778 | 0.8323 |
| `C=0.03` | 0.6608 | 0.6667 | 0.7749 | 0.8315 |
| baseline + temperature | 0.6579 | 0.6637 | 0.7632 | 0.8285 |
| baseline | 0.6520 | 0.6637 | 0.7719 | 0.8275 |

`C=0.1` with temperature has the best source-threshold fusion accuracy, `0.6842`, essentially matching the physics-guided operating point (`0.6813`). However, it still predicts too many target images as generated, so this is not yet a full calibration solution.

## Coefficients

Mean standardized CLIP coefficients:

| fusion config | CLIP coefficient | ConvNeXt coefficient | physics-guided coefficient |
| --- | ---: | ---: | ---: |
| baseline | 2.7024 | 0.7300 | 0.2657 |
| `C=0.03` | 0.7727 | 0.4496 | 0.2950 |
| branch dropout, mean fill | 3.1293 | 1.1606 | 0.6558 |

Strong regularization is the cleanest way to reduce CLIP dominance. Branch dropout increases AUC by forcing the fusion head to use more branches, but it does not fix the target fake-call rate.

## Interpretation

This is a useful mixed result:

- **Best ranking:** branch-dropout fusion, `0.8406` AUC.
- **Best fusion calibration:** strongly regularized fusion, `0.2213` Brier and `0.2091` ECE.
- **Best operating point overall:** physics-guided ResNet-18 still wins default accuracy and ECE.
- **Best source-threshold fusion:** `C=0.1` plus temperature, `0.6842` accuracy.

The next SCP-Fusion step should search a utility-aware fusion objective that explicitly penalizes target-like fake overcalling. A reasonable first candidate is a regularized fusion head with source-heldout threshold selection or a validation objective that combines AUC with Brier/ECE and predicted-positive-rate deviation.

## Artifacts

- per-seed metrics: `reports/assets/ms_cocoai_to_ishu_reverse_fusion_regularization_metrics.csv`
- mean metrics: `reports/assets/ms_cocoai_to_ishu_reverse_fusion_regularization_mean_metrics.csv`
- per-seed threshold calibration: `reports/assets/ms_cocoai_to_ishu_reverse_fusion_regularization_thresholds.csv`
- mean threshold calibration: `reports/assets/ms_cocoai_to_ishu_reverse_fusion_regularization_threshold_means.csv`
- per-seed coefficients: `reports/assets/ms_cocoai_to_ishu_reverse_fusion_regularization_coefficients.csv`
- mean coefficients: `reports/assets/ms_cocoai_to_ishu_reverse_fusion_regularization_coefficient_means.csv`
- local run directory: `runs/ms_cocoai_to_ishu_neural_fusion/`
