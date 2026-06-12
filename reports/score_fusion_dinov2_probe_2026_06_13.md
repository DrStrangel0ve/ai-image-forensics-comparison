# SCP-Fusion + DINOv2 Probe

Run date: 2026-06-13

This follow-up adds the three-seed DINOv2-small frozen encoder as a fifth saved-score branch in SCP-Fusion. The hypothesis was that DINOv2 might complement frozen ConvNeXt: ConvNeXt is the strongest existing ranking branch, while DINOv2 has higher target fake recall but lower precision.

Compared methods:

- SCP-Fusion v0: `combined_v3`, ResNet-18, physics-guided fusion, frozen ConvNeXt-Tiny;
- SCP-Fusion + DINOv2: v0 plus frozen DINOv2-small;
- source-calibrated fusion: previous four-branch source-calibrated model;
- DINOv2 source-calibrated fusion: five-branch fusion with class-balanced source temperature scaling.

## Alignment Note

The first fifth-branch attempt used the seed-7 source feature files for all seeds and exposed a bad alignment symptom: seeds 17/29 had only 22/20 overlapping source rows after joining all branches. The checked-in result below uses seed-specific source feature and ResNet predictions for seeds 17 and 29, restoring 114 aligned source rows per seed.

That caveat matters for the paper: score-level fusion needs split-aware source predictions, not just method labels.

## Cross-Dataset Calibration Summary

Means are over seeds 7, 17, and 29 on the 1,000-image source-balanced MS COCOAI target split.

| method | accuracy | AUC | Brier | ECE | predicted fake rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| DINOv2 source-calibrated fusion | 0.6127 | 0.7502 | 0.3062 | 0.2938 | 0.1727 |
| source-calibrated fusion | 0.6073 | 0.7242 | 0.3123 | 0.2947 | 0.1747 |
| SCP-Fusion + DINOv2 | 0.5983 | 0.7503 | 0.3144 | 0.3069 | 0.1437 |
| SCP-Fusion v0 | 0.5910 | 0.7282 | 0.3190 | 0.3087 | 0.1383 |

Adding DINOv2 gives the strongest ranking result so far: mean AUC rises from `0.7282` to `0.7503`. Source calibration then trades a tiny amount of AUC for the best accuracy, Brier score, and ECE in this comparison.

## Coefficients

Mean logistic-fusion standardized coefficients:

| method | combined_v3 | ResNet-18 | physics-guided | ConvNeXt | DINOv2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SCP-Fusion + DINOv2 | 0.3909 | 0.4542 | 0.6136 | 1.4687 | 0.4353 |
| DINOv2 source-calibrated fusion | 0.3566 | 0.4244 | 0.5216 | 1.2250 | 0.5316 |

ConvNeXt remains the largest branch, but DINOv2 receives a positive coefficient in every seed and becomes slightly stronger under source-calibrated fusion. That supports keeping DINOv2 as a complementary branch rather than treating it as a standalone replacement.

## Source-Heldout Calibration

Each row holds out one generated MS COCOAI source and fits class-balanced temperature scaling on the remaining sources plus real calibration rows.

| method | calibrated accuracy | calibrated AUC | calibrated Brier | calibrated ECE | real FPR | fake detection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DINOv2 source-calibrated fusion | 0.7501 | 0.7386 | 0.1914 | 0.1295 | 0.0640 | 0.2853 |
| SCP-Fusion + DINOv2 | 0.7491 | 0.7371 | 0.1947 | 0.1301 | 0.0480 | 0.2420 |
| source-calibrated fusion | 0.7415 | 0.7082 | 0.1977 | 0.1268 | 0.0747 | 0.2820 |
| SCP-Fusion v0 | 0.7417 | 0.7123 | 0.1995 | 0.1331 | 0.0533 | 0.2293 |

DINOv2 improves the source-heldout ranking and Brier score story. The calibrated DINOv2 fusion has the best held-out fake detection and accuracy, while raw DINOv2 fusion has the lowest real-image FPR.

## Source-Heldout Triage

At the strict 5% calibration error budget:

| method | coverage | triage accuracy | real FPR | fake false clearance | fake detection |
| --- | ---: | ---: | ---: | ---: | ---: |
| DINOv2 source-calibrated fusion | 0.2562 | 0.8016 | 0.0493 | 0.0520 | 0.2513 |
| SCP-Fusion + DINOv2 | 0.2568 | 0.8001 | 0.0493 | 0.0527 | 0.2513 |
| source-calibrated fusion | 0.1975 | 0.7476 | 0.0453 | 0.0540 | 0.2227 |
| SCP-Fusion v0 | 0.2143 | 0.7476 | 0.0520 | 0.0527 | 0.2467 |

At the 10% calibration error budget:

| method | coverage | triage accuracy | real FPR | fake false clearance | fake detection |
| --- | ---: | ---: | ---: | ---: | ---: |
| DINOv2 source-calibrated fusion | 0.4743 | 0.7499 | 0.1240 | 0.1027 | 0.4053 |
| SCP-Fusion + DINOv2 | 0.4867 | 0.7478 | 0.1293 | 0.1033 | 0.4227 |
| SCP-Fusion v0 | 0.4373 | 0.7160 | 0.1307 | 0.1060 | 0.4093 |
| source-calibrated fusion | 0.4309 | 0.7001 | 0.1387 | 0.1033 | 0.4047 |

This is the strongest practical triage result so far. DINOv2 raises both coverage and decided-case accuracy in the source-heldout setting, especially at the strict 5% budget.

## Interpretation

This is a meaningful SCP-Fusion upgrade:

- DINOv2 is not the strongest standalone foundation model, but it adds complementary ranking signal.
- Five-branch SCP-Fusion has the best mean transfer AUC observed so far.
- Five-branch source-calibrated fusion has the best mean target accuracy/Brier/ECE among these fusion variants.
- Source-heldout triage improves substantially, moving from about 21% coverage / 75% triage accuracy to about 26% coverage / 80% triage accuracy at the 5% budget.

The next step is to regenerate publication figures with this result and then test CLIP ViT-B/32 as the other foundation branch.

## Artifacts

Checked-in compact assets:

- `reports/assets/score_fusion_dinov2_calibration_summary.csv`
- `reports/assets/score_fusion_dinov2_calibration_metrics.csv`
- `reports/assets/score_fusion_dinov2_coefficient_summary.csv`
- `reports/assets/score_fusion_dinov2_source_holdout_calibration_summary.csv`
- `reports/assets/score_fusion_dinov2_source_holdout_triage_5pct.csv`
- `reports/assets/score_fusion_dinov2_source_holdout_triage_10pct.csv`

Local run folders are ignored by Git and can be regenerated under:

- `runs/score_fusion_dinov2_aligned/`
- `runs/score_fusion_dinov2_source_calibrated/`
- `runs/calibration_diagnostics/score_fusion_dinov2_compare/`
- `runs/source_holdout_calibration/score_fusion_dinov2_compare/`
- `runs/source_holdout_triage/score_fusion_dinov2_compare_5pct/`
- `runs/source_holdout_triage/score_fusion_dinov2_compare_10pct/`
