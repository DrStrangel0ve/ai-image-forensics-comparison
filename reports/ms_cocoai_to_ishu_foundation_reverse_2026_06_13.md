# MS COCOAI to Ishu Foundation Reverse Transfer

Run date: 2026-06-13

This follow-up closes the first reverse-direction foundation check from the submission-readiness plan. Earlier CLIP results showed strong Ishu -> MS COCOAI transfer ranking. This run asks whether the same frozen foundation branches still behave well when trained on a generator-balanced MS COCOAI export and evaluated on Ishu.

## Setup

- source dataset: `data/raw/ms_cocoai_2026_validation_source_balanced_100/validation`
- target dataset: `data/raw/ishu_ai_vs_real_2026`
- source split: balanced MS COCOAI validation export, `1,000` images
- target split: deterministic Ishu `test` split from the existing `--val-fraction 0.2` protocol
- encoders: `convnext_tiny`, `clip_vit_b_32`, `dinov2_vits14`
- classifier: class-balanced logistic regression on frozen image embeddings
- seeds: `7`, `17`, `29`
- device: `cuda`

The CLIP loader reports unused text-tower weights when loading `CLIPVisionModel`; this is expected because the run uses only the vision tower.

## Source Validation

Mean over three seeds on held-out MS COCOAI validation rows:

| encoder | accuracy | AUC | Brier | ECE | predicted fake rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| ConvNeXt-Tiny | 0.8850 | 0.9433 | 0.0924 | 0.0718 | 0.4983 |
| CLIP ViT-B/32 | 0.9650 | 0.9956 | 0.0244 | 0.0237 | 0.5183 |
| DINOv2-small | 0.8217 | 0.8960 | 0.1467 | 0.1331 | 0.4983 |

CLIP is dominant on the MS COCOAI source split. That makes the target result below a harder and more useful diagnostic: the problem is not weak source fitting.

## Reverse Target Transfer

Mean over three seeds on MS COCOAI -> Ishu:

| encoder | accuracy | AUC | Brier | ECE | precision | recall | F1 | predicted fake rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ConvNeXt-Tiny | 0.6579 | 0.8081 | 0.2860 | 0.2950 | 0.6031 | 0.8988 | 0.7210 | 0.7339 |
| CLIP ViT-B/32 | 0.6228 | 0.8243 | 0.3317 | 0.3566 | 0.5663 | 0.9940 | 0.7215 | 0.8626 |
| DINOv2-small | 0.6170 | 0.6669 | 0.3472 | 0.3499 | 0.5792 | 0.8095 | 0.6750 | 0.6871 |

CLIP remains the strongest transferred ranker in the reverse direction, but its default threshold over-calls generated images on Ishu. ConvNeXt has the best default-threshold accuracy and better calibration metrics, while CLIP has near-perfect generated-image recall.

## Threshold Calibration

Each seed chooses a clean threshold from the held-out MS COCOAI source predictions and applies it unchanged to the Ishu test split. The oracle threshold is fitted on the Ishu target scores and is included only as a diagnostic upper bound.

| encoder | default accuracy | source-threshold accuracy | oracle accuracy | AUC |
| --- | ---: | ---: | ---: | ---: |
| ConvNeXt-Tiny | 0.6579 | 0.6784 | 0.7602 | 0.8081 |
| CLIP ViT-B/32 | 0.6228 | 0.6462 | 0.7749 | 0.8243 |
| DINOv2-small | 0.6170 | 0.6170 | 0.6637 | 0.6669 |

Source-threshold calibration helps ConvNeXt and CLIP modestly, but the large oracle gap remains. For CLIP, the Ishu oracle thresholds are very high, roughly `0.95` to `0.998`, which explains the default-threshold over-call pattern.

## Interpretation

This reverse direction strengthens the paper claim that ranking, calibration, and binary operating points are separate forensic questions:

- CLIP is the most transferable foundation ranker in both tested directions so far.
- ConvNeXt is currently the better reverse-direction deployment baseline at the default and source-threshold operating points.
- DINOv2-small is weaker in this direction, so it remains a complementary branch rather than a standalone lead.
- The next SCP-Fusion step should not only add more branches; it should explicitly learn source-aware calibration or thresholding from generator/source structure.

## Artifacts

- per-seed metrics: `reports/assets/ms_cocoai_to_ishu_foundation_metrics.csv`
- mean metrics: `reports/assets/ms_cocoai_to_ishu_foundation_mean_metrics.csv`
- per-seed threshold calibration: `reports/assets/ms_cocoai_to_ishu_foundation_thresholds.csv`
- mean threshold calibration: `reports/assets/ms_cocoai_to_ishu_foundation_threshold_means.csv`
- local run directory: `runs/ms_cocoai_to_ishu_foundation/`
