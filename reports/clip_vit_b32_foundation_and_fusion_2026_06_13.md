# CLIP ViT-B/32 Foundation and Fusion Probe

Run date: 2026-06-13

This follow-up completes the CLIP side of the frozen-foundation roadmap and tests whether CLIP should be a standalone detector branch, a fifth SCP-Fusion branch, or part of an all-foundation fusion with DINOv2.

## Setup

- encoder: `clip_vit_b_32`
- weights: `openai/clip-vit-base-patch32`
- classifier: class-balanced logistic regression on frozen image embeddings
- source dataset: Ishu AI-vs-real 2026
- target dataset: source-balanced MS COCOAI validation, `1,000` images
- seeds: `7`, `17`, `29`
- device: `cuda`

The Hugging Face loader reports CLIP text-tower weights as unexpected when loading `CLIPVisionModel`; this is expected because the run uses only the vision tower.

## Standalone CLIP Baseline

| split | mean accuracy | mean AUC | Brier | ECE | predicted fake rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| Ishu validation | 0.9152 | 0.9701 | 0.0728 | 0.0678 | 0.4942 |
| Ishu -> MS COCOAI | 0.6363 | 0.8641 | 0.3146 | 0.3326 | 0.1630 |

CLIP is now the strongest standalone cross-domain ranker in this repo. Its Ishu-to-MS-COCOAI AUC is far above frozen ConvNeXt-Tiny (`0.7139`), DINOv2-small (`0.7063`), and SCP-Fusion + DINOv2 (`0.7503`).

The default threshold remains conservative: on the balanced MS COCOAI target split, CLIP predicts only about 16% of images as generated.

## Target Source Breakdown

Mean over seeds at threshold `0.5`:

| source | n per seed | accuracy | mean fake score | detection / FPR |
| --- | ---: | ---: | ---: | ---: |
| real | 500 | 0.9733 | 0.0333 | FPR 0.0267 |
| SD 2.1 | 100 | 0.2367 | 0.2613 | detect 0.2367 |
| SDXL | 100 | 0.5667 | 0.5538 | detect 0.5667 |
| SD3 | 100 | 0.2800 | 0.3069 | detect 0.2800 |
| DALL-E 3 | 100 | 0.2300 | 0.2428 | detect 0.2300 |
| Midjourney 6 | 100 | 0.1833 | 0.2063 | detect 0.1833 |

The threshold story is visible by source: CLIP is excellent at keeping real images below the fake threshold and separates SDXL best, but it under-calls SD2.1, SD3, DALL-E 3, and Midjourney 6 at the default operating point.

## Score-Fusion Comparison

Mean over three seeds on Ishu -> source-balanced MS COCOAI:

| method | accuracy | AUC | Brier | ECE | predicted fake rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| CLIP standalone | 0.6363 | 0.8641 | 0.3146 | 0.3326 | 0.1630 |
| all-foundation source-calibrated fusion | 0.6267 | 0.7974 | 0.2985 | 0.3089 | 0.1500 |
| CLIP source-calibrated fusion | 0.6287 | 0.7839 | 0.3026 | 0.3114 | 0.1520 |
| SCP-Fusion all-foundation | 0.6163 | 0.7995 | 0.3118 | 0.3263 | 0.1323 |
| SCP-Fusion + CLIP | 0.6183 | 0.7915 | 0.3112 | 0.3253 | 0.1343 |
| DINOv2 source-calibrated fusion | 0.6127 | 0.7502 | 0.3062 | 0.2938 | 0.1727 |
| SCP-Fusion + DINOv2 | 0.5983 | 0.7503 | 0.3144 | 0.3069 | 0.1437 |
| SCP-Fusion v0 | 0.5910 | 0.7282 | 0.3190 | 0.3087 | 0.1383 |

Adding CLIP clearly improves the fusion family: all-foundation SCP-Fusion reaches `0.7995` AUC versus `0.7282` for v0 and `0.7503` for the DINOv2-only extension. But fusion still does not beat standalone CLIP ranking.

Mean standardized fusion coefficients show the meta-classifier is using CLIP heavily:

| family | top coefficient | mean standardized coefficient |
| --- | --- | ---: |
| SCP-Fusion + CLIP | CLIP ViT-B/32 | 1.5069 |
| SCP-Fusion all-foundation | CLIP ViT-B/32 | 1.4578 |
| CLIP source-calibrated fusion | CLIP ViT-B/32 | 1.4242 |
| all-foundation source-calibrated fusion | CLIP ViT-B/32 | 1.3796 |

That makes this a useful negative/mixed result: CLIP is the dominant branch, but score-level fusion trained only on the small Ishu validation split appears to compress or distort CLIP's target ranking.

## Source-Heldout Calibration

Each row holds out one generated MS COCOAI source and fits class-balanced temperature scaling on the remaining generated sources plus real calibration rows.

| method | calibrated accuracy | calibrated AUC | calibrated Brier | calibrated ECE | real FPR | fake detection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CLIP standalone | 0.7855 | 0.8705 | 0.1508 | 0.1396 | 0.0200 | 0.2993 |
| all-foundation source-calibrated fusion | 0.7810 | 0.7919 | 0.1758 | 0.1367 | 0.0173 | 0.2767 |
| CLIP source-calibrated fusion | 0.7783 | 0.7749 | 0.1791 | 0.1399 | 0.0227 | 0.2807 |
| SCP-Fusion all-foundation | 0.7768 | 0.7925 | 0.1812 | 0.1529 | 0.0120 | 0.2487 |
| SCP-Fusion + CLIP | 0.7779 | 0.7832 | 0.1825 | 0.1499 | 0.0120 | 0.2527 |
| DINOv2 source-calibrated fusion | 0.7501 | 0.7386 | 0.1914 | 0.1295 | 0.0640 | 0.2853 |
| SCP-Fusion + DINOv2 | 0.7491 | 0.7371 | 0.1947 | 0.1301 | 0.0480 | 0.2420 |
| SCP-Fusion v0 | 0.7417 | 0.7123 | 0.1995 | 0.1331 | 0.0533 | 0.2293 |

CLIP has the best held-out-source ranking, accuracy, and Brier score. All-foundation source-calibrated fusion has the lowest ECE and almost the lowest real FPR, so it remains useful for conservative forensic review, but CLIP is the stronger single branch on this target.

## Source-Heldout Triage

At the strict 5% calibration error budget:

| method | coverage | triage accuracy | real FPR | fake false clearance | fake detection |
| --- | ---: | ---: | ---: | ---: | ---: |
| CLIP standalone | 0.4747 | 0.9261 | 0.0267 | 0.0540 | 0.3580 |
| SCP-Fusion all-foundation | 0.3029 | 0.8733 | 0.0333 | 0.0500 | 0.3380 |
| all-foundation source-calibrated fusion | 0.2994 | 0.8721 | 0.0320 | 0.0513 | 0.3560 |
| SCP-Fusion + CLIP | 0.2888 | 0.8522 | 0.0373 | 0.0527 | 0.3467 |
| CLIP source-calibrated fusion | 0.2705 | 0.8429 | 0.0360 | 0.0553 | 0.3567 |
| DINOv2 source-calibrated fusion | 0.2562 | 0.8016 | 0.0493 | 0.0520 | 0.2513 |
| SCP-Fusion v0 | 0.2143 | 0.7476 | 0.0520 | 0.0527 | 0.2467 |

At the 10% calibration error budget:

| method | coverage | triage accuracy | real FPR | fake false clearance | fake detection |
| --- | ---: | ---: | ---: | ---: | ---: |
| CLIP standalone | 0.6878 | 0.8792 | 0.0747 | 0.1040 | 0.5927 |
| SCP-Fusion all-foundation | 0.5215 | 0.8165 | 0.0907 | 0.1073 | 0.4993 |
| all-foundation source-calibrated fusion | 0.5206 | 0.8116 | 0.0947 | 0.1060 | 0.4947 |
| SCP-Fusion + CLIP | 0.4817 | 0.7990 | 0.0947 | 0.0993 | 0.5007 |
| CLIP source-calibrated fusion | 0.4840 | 0.7766 | 0.1093 | 0.1033 | 0.4947 |
| DINOv2 source-calibrated fusion | 0.4743 | 0.7499 | 0.1240 | 0.1027 | 0.4053 |
| SCP-Fusion v0 | 0.4373 | 0.7160 | 0.1307 | 0.1060 | 0.4093 |

This is the strongest triage result in the project so far. Under source-heldout thresholds, CLIP standalone decides on nearly half the target images at about 93% decided-case accuracy under the strict 5% budget.

## Interpretation

CLIP changes the paper story:

- Frozen foundation baselines should be treated as first-class baselines, not just optional feature branches.
- CLIP ViT-B/32 is currently the strongest cross-domain ranker and triage model in this repo.
- All-foundation SCP-Fusion improves over previous fusion variants but still trails standalone CLIP on ranking and triage.
- Score-level fusion trained on small source validation splits may overfit source correlations and suppress the strongest branch under generator shift.
- The next SCP-Fusion iteration should consider calibration-aware or source-heldout fusion training rather than simply adding more branches.

## Artifacts

Checked-in compact assets:

- `reports/assets/clip_vit_b32_three_seed_metrics.csv`
- `reports/assets/clip_vit_b32_ms_cocoai_source_means.csv`
- `reports/assets/score_fusion_clip_calibration_summary.csv`
- `reports/assets/score_fusion_clip_calibration_metrics.csv`
- `reports/assets/score_fusion_clip_coefficient_summary.csv`
- `reports/assets/score_fusion_clip_coefficients.csv`
- `reports/assets/score_fusion_clip_source_holdout_calibration_summary.csv`
- `reports/assets/score_fusion_clip_source_holdout_triage_5pct.csv`
- `reports/assets/score_fusion_clip_source_holdout_triage_10pct.csv`

Local run folders are ignored by Git and can be regenerated under:

- `runs/ishu_ai_vs_real_2026_frozen_encoder/clip_vit_b_32_seed{7,17,29}/`
- `runs/ishu_to_ms_cocoai_source_balanced_seed{7,17,29}/clip_vit_b_32_frozen/`
- `runs/score_fusion_clip_aligned/`
- `runs/score_fusion_clip_source_calibrated/`
- `runs/score_fusion_foundation_aligned/`
- `runs/score_fusion_foundation_source_calibrated/`
- `runs/calibration_diagnostics/score_fusion_clip_compare/`
- `runs/source_holdout_calibration/score_fusion_clip_compare/`
- `runs/source_holdout_triage/score_fusion_clip_compare_5pct/`
- `runs/source_holdout_triage/score_fusion_clip_compare_10pct/`
