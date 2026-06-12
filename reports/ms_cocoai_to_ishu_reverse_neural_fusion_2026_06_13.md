# MS COCOAI to Ishu Reverse Neural and Fusion Baselines

Run date: 2026-06-13

This follow-up extends the reverse foundation-transfer check to the non-foundation branches: `combined_v3`, vanilla ResNet-18, physics-guided ResNet-18, and an all-branch score fusion over conventional, neural, physics-guided, ConvNeXt, CLIP, and DINOv2 scores.

## Setup

- source dataset: `data/raw/ms_cocoai_2026_validation_source_balanced_100/validation`
- target dataset: `data/raw/ishu_ai_vs_real_2026`
- source split: deterministic `80/20` split with `--val-fraction 0.2`
- target split: deterministic Ishu `test` split using the same seed and `--val-fraction 0.2`
- seeds: `7`, `17`, `29`
- conventional branch: `combined_v3`, logistic regression, `128` px feature extraction
- neural branch: ImageNet-pretrained ResNet-18, `6` epochs, `128` px inputs
- physics-guided branch: ImageNet-pretrained ResNet-18 fused with `combined_v3`, `6` epochs
- fusion branches: `combined_v3`, ResNet-18, physics-guided ResNet-18, ConvNeXt-Tiny, CLIP ViT-B/32, DINOv2-small
- device: `cuda`

## Reverse Target Results

Mean over three seeds on MS COCOAI -> Ishu:

| method | accuracy | AUC | Brier | ECE | precision | recall | predicted fake rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| all-branch fusion + temperature | 0.6579 | 0.8285 | 0.3067 | 0.3304 | 0.5930 | 0.9762 | 0.8099 |
| all-branch fusion | 0.6520 | 0.8275 | 0.2944 | 0.3181 | 0.5876 | 0.9821 | 0.8216 |
| CLIP ViT-B/32 | 0.6228 | 0.8243 | 0.3317 | 0.3566 | 0.5663 | 0.9940 | 0.8626 |
| ConvNeXt-Tiny | 0.6579 | 0.8081 | 0.2860 | 0.2950 | 0.6031 | 0.8988 | 0.7339 |
| physics-guided ResNet-18 + `combined_v3` | 0.6871 | 0.7420 | 0.2436 | 0.1902 | 0.6687 | 0.7202 | 0.5292 |
| ResNet-18 | 0.6404 | 0.7005 | 0.2877 | 0.2671 | 0.6151 | 0.7262 | 0.5819 |
| DINOv2-small | 0.6170 | 0.6669 | 0.3472 | 0.3499 | 0.5792 | 0.8095 | 0.6871 |
| `combined_v3` | 0.5468 | 0.5772 | 0.3078 | 0.2472 | 0.5288 | 0.7262 | 0.6754 |

The result splits the story cleanly:

- the all-branch score fusion has the best mean AUC, slightly ahead of standalone CLIP;
- physics-guided ResNet-18 has the best default-threshold accuracy, Brier score, ECE, and much more plausible fake-call rate;
- `combined_v3` alone remains too weak for this reverse direction, but it helps as part of the physics-guided branch.

## Source-Threshold Calibration

Thresholds are selected on the held-out MS COCOAI source predictions and then applied unchanged to Ishu. Oracle thresholds use target labels and are diagnostic only.

| method | default accuracy | source-threshold accuracy | oracle accuracy | AUC |
| --- | ---: | ---: | ---: | ---: |
| all-branch fusion + temperature | 0.6579 | 0.6637 | 0.7632 | 0.8285 |
| all-branch fusion | 0.6520 | 0.6637 | 0.7719 | 0.8275 |
| CLIP ViT-B/32 | 0.6228 | 0.6462 | 0.7749 | 0.8243 |
| ConvNeXt-Tiny | 0.6579 | 0.6784 | 0.7602 | 0.8081 |
| physics-guided ResNet-18 + `combined_v3` | 0.6871 | 0.6813 | 0.7105 | 0.7420 |
| ResNet-18 | 0.6404 | 0.6491 | 0.6784 | 0.7005 |
| DINOv2-small | 0.6170 | 0.6170 | 0.6637 | 0.6669 |
| `combined_v3` | 0.5468 | 0.5409 | 0.5877 | 0.5772 |

Source-threshold calibration does not close the gap for all-branch fusion. It still over-calls generated images on Ishu because the fusion head is dominated by CLIP-like ranking behavior.

## Fusion Weights

Mean standardized coefficients over seeds:

| fusion | top branch | mean standardized coefficient |
| --- | --- | ---: |
| all-branch fusion | CLIP ViT-B/32 | 2.7024 |
| all-branch fusion + temperature | CLIP ViT-B/32 | 2.2924 |
| all-branch fusion + temperature | ConvNeXt-Tiny | 0.7459 |
| all-branch fusion | ConvNeXt-Tiny | 0.7300 |

The score fusion mostly becomes a CLIP-led ranker with ConvNeXt support. That explains why fusion improves AUC slightly but does not inherit the physics-guided branch's operating-point behavior.

## Interpretation

This run fills the reverse-direction paper gap. It supports a stronger and more nuanced SCP-Fusion claim:

- frozen CLIP provides the best transferable ranking signal;
- physics-guided ResNet-18 provides the best reverse-direction calibrated operating point;
- naive score fusion can improve AUC but still needs source-aware calibration or utility-aware thresholding before it is a good detector;
- future SCP-Fusion work should train the fusion objective against source-heldout operating points, not just source-domain log-loss.

## Artifacts

- per-seed metrics: `reports/assets/ms_cocoai_to_ishu_reverse_all_methods_metrics.csv`
- mean metrics: `reports/assets/ms_cocoai_to_ishu_reverse_all_methods_mean_metrics.csv`
- per-seed threshold calibration: `reports/assets/ms_cocoai_to_ishu_reverse_all_methods_thresholds.csv`
- mean threshold calibration: `reports/assets/ms_cocoai_to_ishu_reverse_all_methods_threshold_means.csv`
- score-fusion coefficients: `reports/assets/ms_cocoai_to_ishu_reverse_score_fusion_coefficients.csv`
- score-fusion coefficient means: `reports/assets/ms_cocoai_to_ishu_reverse_score_fusion_coefficient_means.csv`
- local run directory: `runs/ms_cocoai_to_ishu_neural_fusion/`
