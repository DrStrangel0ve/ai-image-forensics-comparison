# Defactify / MS COCOAI Robustness Variants

Run date: 2026-06-12

This run evaluates whether the strongest conventional baseline, `combined_v3`, and the saved ResNet-18 baseline survive common image-sharing transformations. The test uses the same 1,000-image source-balanced Defactify/MS COCOAI validation slice from `reports/ms_cocoai_source_balanced_validation.md`.

The variants are motivated by robust-detection datasets such as NTIRE 2026, where in-the-wild transformations are part of the detection problem: https://huggingface.co/datasets/deepfakesMSU/NTIRE-RobustAIGenDetection-train

## Variants

| variant | transform |
| --- | --- |
| `jpeg70` | JPEG recompression at quality 70, then saved to the project standard JPEG output. |
| `blur1` | Gaussian blur with radius 1.0. |
| `resize_half` | Bicubic downscale to half resolution, then upsample back to the original size. |
| `crop85` | Center crop to 85% width/height, then resize back to the original size. |

## Results

Deltas are relative to each method's clean source-balanced validation score.

| variant | method | accuracy | accuracy delta | roc_auc | roc_auc delta |
| --- | --- | ---: | ---: | ---: | ---: |
| blur1 | combined_v3 | 0.6350 | -0.0970 | 0.7665 | -0.0362 |
| blur1 | resnet18 | 0.7990 | -0.0170 | 0.8879 | -0.0089 |
| crop85 | combined_v3 | 0.7140 | -0.0180 | 0.7977 | -0.0049 |
| crop85 | resnet18 | 0.8100 | -0.0060 | 0.8938 | -0.0029 |
| jpeg70 | combined_v3 | 0.7340 | +0.0020 | 0.7994 | -0.0033 |
| jpeg70 | resnet18 | 0.8170 | +0.0010 | 0.8974 | +0.0007 |
| resize_half | combined_v3 | 0.6940 | -0.0380 | 0.7920 | -0.0107 |
| resize_half | resnet18 | 0.8110 | -0.0050 | 0.8938 | -0.0030 |

## Interpretation

JPEG recompression at quality 70 barely changes either detector, which is a useful sanity check because `combined_v3` explicitly includes recompression features.

Gaussian blur is the main weakness for the conventional baseline. `combined_v3` drops 9.7 accuracy points under `blur1`, while ResNet-18 drops 1.7 points. Half-resolution roundtrip also hurts `combined_v3` more than ResNet-18, although the AUC drop is modest. Center crop/resize is comparatively mild for both methods.

This suggests the next conventional improvement should focus on blur/resize-normalized residual features, or on reporting AUC/threshold-calibrated metrics separately from the default 0.5 threshold under transformed imagery.

## Reproduce

```powershell
python scripts/make_robustness_variants.py `
  --data-dir data/raw/ms_cocoai_2026_validation_source_balanced_100 `
  --out-dir data/raw/ms_cocoai_2026_source_balanced_robustness_variants `
  --variants jpeg70 blur1 resize_half crop85 `
  --format jpg
```

Evaluate each variant with the saved `combined_v3` and ResNet-18 models:

```powershell
python scripts/evaluate_feature_model.py `
  --model-dir runs/ms_cocoai_2026_subset_500/feature_combined_v3_logistic_regression `
  --target-dir data/raw/ms_cocoai_2026_source_balanced_robustness_variants/jpeg70 `
  --output-dir runs/ms_cocoai_2026_robustness_eval/jpeg70/combined_v3 `
  --image-size 128 `
  --target-split all

python scripts/evaluate_neural_net.py `
  --model-dir runs/ms_cocoai_2026_subset_500/resnet18 `
  --target-dir data/raw/ms_cocoai_2026_source_balanced_robustness_variants/jpeg70 `
  --output-dir runs/ms_cocoai_2026_robustness_eval/jpeg70/resnet18 `
  --image-size 128 `
  --batch-size 64 `
  --num-workers 0 `
  --device cuda `
  --target-split all
```

Repeat the two evaluation commands for `blur1`, `resize_half`, and `crop85`, then summarize:

```powershell
python scripts/summarize_robustness_eval.py `
  --out-dir runs/ms_cocoai_2026_robustness_eval/summary `
  --baseline combined_v3=runs/ms_cocoai_2026_source_balanced_eval/combined_v3/metrics.json `
  --baseline resnet18=runs/ms_cocoai_2026_source_balanced_eval/resnet18/metrics.json `
  --metrics jpeg70:combined_v3=runs/ms_cocoai_2026_robustness_eval/jpeg70/combined_v3/metrics.json `
  --metrics jpeg70:resnet18=runs/ms_cocoai_2026_robustness_eval/jpeg70/resnet18/metrics.json `
  --metrics blur1:combined_v3=runs/ms_cocoai_2026_robustness_eval/blur1/combined_v3/metrics.json `
  --metrics blur1:resnet18=runs/ms_cocoai_2026_robustness_eval/blur1/resnet18/metrics.json `
  --metrics resize_half:combined_v3=runs/ms_cocoai_2026_robustness_eval/resize_half/combined_v3/metrics.json `
  --metrics resize_half:resnet18=runs/ms_cocoai_2026_robustness_eval/resize_half/resnet18/metrics.json `
  --metrics crop85:combined_v3=runs/ms_cocoai_2026_robustness_eval/crop85/combined_v3/metrics.json `
  --metrics crop85:resnet18=runs/ms_cocoai_2026_robustness_eval/crop85/resnet18/metrics.json
```
