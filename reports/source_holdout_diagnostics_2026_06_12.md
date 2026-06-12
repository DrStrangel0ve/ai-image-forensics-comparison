# Source-Heldout Diagnostics

Run date: 2026-06-12

This report adds a stricter diagnostic for the Ishu -> Defactify/MS COCOAI transfer runs. For each generated MS COCOAI source, the new `scripts/summarize_source_holdout.py` script:

1. holds out one generated source;
2. selects a threshold on all other generated sources plus a deterministic calibration subset of real images;
3. evaluates on the held-out generated source plus the remaining real images.

This is not a full leave-one-generator-out training run, because it uses existing prediction scores. It is a fast calibration and score-separation diagnostic that exposes whether a method can transfer thresholds across generators.

## Three-Seed Method Summary

| method | mean default accuracy | mean source-heldout threshold accuracy | mean oracle accuracy | mean AUC |
| --- | ---: | ---: | ---: | ---: |
| `combined_v3` | 0.6802 | 0.3244 | 0.7272 | 0.5614 |
| physics-guided ResNet-18 + `combined_v3` | 0.6971 | 0.4840 | 0.7411 | 0.6481 |
| ResNet-18 | 0.6796 | 0.4206 | 0.7341 | 0.6308 |

The default-threshold result keeps the earlier story: physics-guided fusion has the strongest three-seed mean, but only slightly. The new source-heldout threshold result is more sobering. Selecting a threshold from the other fake sources does not generalize cleanly to the held-out source/real mixture. It often pushes the threshold extremely low and raises real-image false positives.

## Per-Seed Summary

| seed | method | mean default accuracy | mean source-heldout threshold accuracy | mean oracle accuracy | mean AUC |
| ---: | --- | ---: | ---: | ---: | ---: |
| 7 | physics-guided fusion | 0.6594 | 0.4783 | 0.7360 | 0.6471 |
| 7 | ResNet-18 | 0.6794 | 0.4491 | 0.7309 | 0.6340 |
| 7 | `combined_v3` | 0.6806 | 0.3091 | 0.7251 | 0.5633 |
| 17 | ResNet-18 | 0.7097 | 0.4949 | 0.7417 | 0.6577 |
| 17 | physics-guided fusion | 0.6863 | 0.4829 | 0.7240 | 0.6036 |
| 17 | `combined_v3` | 0.6709 | 0.3109 | 0.7257 | 0.5471 |
| 29 | physics-guided fusion | 0.7457 | 0.4909 | 0.7634 | 0.6935 |
| 29 | `combined_v3` | 0.6891 | 0.3531 | 0.7309 | 0.5739 |
| 29 | ResNet-18 | 0.6497 | 0.3177 | 0.7297 | 0.6006 |

## Interpretation

The diagnostic explains why simple source calibration is not enough. In all three seeds, source-heldout threshold selection frequently chooses thresholds near zero. That improves held-out fake detection, but it also labels too many real MS COCO images as generated.

Examples:

- Seed 17 ResNet-18 source-heldout thresholds give real false-positive rates around 0.58 to 0.68.
- Seed 17 physics-guided fusion improves held-out fake detection on SD2.1, SDXL, and DALL-E 3 relative to `combined_v3`, but still has real false-positive rates around 0.60 to 0.69.
- Seed 29 physics-guided fusion has the best default and oracle ranking, but source-heldout thresholds still produce real false-positive rates around 0.61 to 0.66.
- Standalone `combined_v3` is especially unstable: several source-heldout thresholds are effectively zero, causing real false-positive rates near 0.99.

So the fusion model is still useful, but the next version needs better score geometry, not just better threshold choice. The SCP-Fusion roadmap should prioritize branch diversity and source-aware training:

- add foundation embeddings;
- add multiscale spectral/noise `combined_v4`;
- add reconstruction-error features;
- train or validate with leave-one-generator/source groups;
- report calibration metrics such as Brier score and ECE in addition to AUC and accuracy.

## Reproduce

Seed 17 example:

```powershell
python scripts/summarize_source_holdout.py `
  --metadata data\raw\ms_cocoai_2026_validation_source_balanced_100\metadata.csv `
  --out-dir runs\source_holdout\ishu_to_ms_cocoai_seed17 `
  --split validation `
  --seed 17 `
  --predictions combined_v3=runs\ishu_to_ms_cocoai_source_balanced_seed17\combined_v3\predictions.csv `
  --predictions resnet18=runs\ishu_to_ms_cocoai_source_balanced_seed17\resnet18\predictions.csv `
  --predictions physics_guided=runs\ishu_to_ms_cocoai_source_balanced_seed17\physics_guided_resnet18_combined_v3\predictions.csv
```

The same command was run for seeds 7, 17, and 29, changing the seed and prediction directories.
