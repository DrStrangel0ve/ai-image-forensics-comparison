# MS COCOAI to Ishu Source-Threshold Fusion Probe

Run date: 2026-06-13

This probe turns the reverse-fusion calibration lesson into a reusable experiment path. `scripts/fuse_prediction_scores.py` can now reserve a deterministic class-balanced source slice for threshold selection and evaluate the selected operating point directly, instead of requiring a separate post-hoc threshold summarizer.

## Setup

- source dataset: `data/raw/ms_cocoai_2026_validation_source_balanced_100/validation`
- target dataset: `data/raw/ishu_ai_vs_real_2026`
- seeds: `7`, `17`, `29`
- fusion branches: `combined_v3`, ResNet-18, physics-guided ResNet-18, ConvNeXt-Tiny, CLIP ViT-B/32, DINOv2-small
- fusion fit rows: 80% of aligned source rows
- threshold rows: held-out 20% of aligned source rows
- threshold policy: `--threshold-strategy source_accuracy`

## Reverse Target Results

Mean over three seeds on MS COCOAI -> Ishu:

| fusion config | accuracy | AUC | Brier | ECE | predicted fake rate | selected threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `C=0.03` + source threshold | 0.6959 | 0.8291 | 0.2188 | 0.2060 | 0.7076 | 0.5880 |
| branch dropout + source threshold | 0.6725 | 0.8341 | 0.3083 | 0.3299 | 0.7836 | 0.6120 |
| `C=0.1` + temperature + source threshold | 0.6754 | 0.7975 | 0.3066 | 0.3267 | 0.7749 | 0.6103 |
| baseline + source threshold | 0.6608 | 0.8215 | 0.2984 | 0.3217 | 0.8129 | 0.6012 |

The `C=0.03` source-threshold fusion is the first reverse all-branch fusion setting to beat the physics-guided ResNet-18 default accuracy (`0.6959` vs `0.6871`) while preserving a much stronger ranking signal (`0.8291` vs `0.7420` AUC). It also improves Brier score relative to physics-guided (`0.2188` vs `0.2436`), although physics-guided still has slightly better ECE (`0.1902` vs `0.2060`) and a more plausible fake-call rate.

Follow-up: `reports/ms_cocoai_to_ishu_threshold_tiebreak_probe_2026_06_13.md` keeps the same score model and source objective but uses conservative threshold tie-breaking. That raises the reverse operating-point accuracy to `0.7193` and reduces the predicted fake rate to `0.6608`.

## Interpretation

This supports a sharper SCP-Fusion framing:

- strong regularization reduces CLIP dominance enough for source-threshold selection to matter;
- branch dropout remains the best ranking-oriented fusion family, but its operating point is still too fake-heavy;
- held-out source thresholding is a useful bridge between naive score fusion and a future utility-aware training objective;
- the next method step should optimize a combined objective over ranking, Brier/ECE, and predicted-positive-rate deviation.

## Reproduction

Example seed/config:

```powershell
.\.venv\Scripts\python scripts\fuse_prediction_scores.py `
  --out-dir runs\ms_cocoai_to_ishu_neural_fusion\score_fusion_all6_c003_source_acc_seed7 `
  --seed 7 `
  --fusion-c 0.03 `
  --calibration-fraction 0.2 `
  --threshold-strategy source_accuracy `
  --train combined_v3=runs\ms_cocoai_to_ishu_neural_fusion\combined_v3_seed7\predictions.csv `
  --train resnet18=runs\ms_cocoai_to_ishu_neural_fusion\resnet18_seed7\predictions.csv `
  --train physics_guided=runs\ms_cocoai_to_ishu_neural_fusion\physics_guided_resnet18_combined_v3_seed7\predictions.csv `
  --train convnext_tiny=runs\ms_cocoai_to_ishu_foundation\convnext_tiny_seed7\predictions.csv `
  --train clip_vit_b_32=runs\ms_cocoai_to_ishu_foundation\clip_vit_b_32_seed7\predictions.csv `
  --train dinov2_vits14=runs\ms_cocoai_to_ishu_foundation\dinov2_vits14_seed7\predictions.csv `
  --variant ishu_test:combined_v3=runs\ms_cocoai_to_ishu_neural_fusion\combined_v3_seed7_to_ishu_test\predictions.csv `
  --variant ishu_test:resnet18=runs\ms_cocoai_to_ishu_neural_fusion\resnet18_seed7_to_ishu_test\predictions.csv `
  --variant ishu_test:physics_guided=runs\ms_cocoai_to_ishu_neural_fusion\physics_guided_resnet18_combined_v3_seed7_to_ishu_test\predictions.csv `
  --variant ishu_test:convnext_tiny=runs\ms_cocoai_to_ishu_foundation\convnext_tiny_seed7_to_ishu_test\predictions.csv `
  --variant ishu_test:clip_vit_b_32=runs\ms_cocoai_to_ishu_foundation\clip_vit_b_32_seed7_to_ishu_test\predictions.csv `
  --variant ishu_test:dinov2_vits14=runs\ms_cocoai_to_ishu_foundation\dinov2_vits14_seed7_to_ishu_test\predictions.csv
```

## Artifacts

- per-seed metrics: `reports/assets/ms_cocoai_to_ishu_source_threshold_fusion_metrics.csv`
- mean metrics: `reports/assets/ms_cocoai_to_ishu_source_threshold_fusion_mean_metrics.csv`
- local run directory: `runs/ms_cocoai_to_ishu_neural_fusion/`
