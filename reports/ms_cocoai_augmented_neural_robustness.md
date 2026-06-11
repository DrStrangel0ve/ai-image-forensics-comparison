# Defactify / MS COCOAI Augmented Neural Robustness

Run date: 2026-06-12

This follow-up adds the same deterministic robustness variants to the neural baseline that were already used for the augmented `combined_v3` conventional baseline:

- `jpeg70`
- `blur1`
- `resize_half`
- `crop85`

The neural trainer now accepts `--train-augment-variants`, and `scripts/run_benchmark.py` forwards the same setting through `--neural-train-augment-variants`. The augmentation is applied in memory with `src/forensic_compare/transforms.py`; no duplicate training dataset has to be written to disk.

The ResNet-18 comparison uses the existing 1,000-image Defactify/MS COCOAI training subset and source-balanced 1,000-image validation slice: 500 real images and 500 generated images balanced across SD2.1, SDXL, SD3, DALL-E 3, and Midjourney v6.

## Clean Validation

| method | accuracy | precision | recall | f1 | roc_auc | train images seen |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| combined_v3 | 0.7320 | 0.7248 | 0.7480 | 0.7362 | 0.8027 | 1000 |
| combined_v3_augmented | 0.7320 | 0.7417 | 0.7120 | 0.7265 | 0.7984 | 5000 |
| resnet18 | 0.8160 | 0.8098 | 0.8260 | 0.8178 | 0.8967 | 1000 |
| resnet18_augmented | 0.8070 | 0.8699 | 0.7220 | 0.7891 | 0.8879 | 5000 |

The augmented ResNet-18 selected epoch 3 on its internal validation split, with 0.8090 internal accuracy and 0.8892 internal AUC. On the separate source-balanced validation slice, it is 0.9 accuracy points and 0.0088 AUC below the original ResNet-18.

## Robustness Results

Deltas are relative to each method's own clean source-balanced validation score.

| variant | method | accuracy | accuracy delta | roc_auc | roc_auc delta |
| --- | --- | ---: | ---: | ---: | ---: |
| blur1 | combined_v3 | 0.6350 | -0.0970 | 0.7665 | -0.0362 |
| blur1 | combined_v3_augmented | 0.6980 | -0.0340 | 0.7864 | -0.0119 |
| blur1 | resnet18 | 0.7990 | -0.0170 | 0.8879 | -0.0089 |
| blur1 | resnet18_augmented | 0.8030 | -0.0040 | 0.8828 | -0.0051 |
| crop85 | combined_v3 | 0.7140 | -0.0180 | 0.7977 | -0.0049 |
| crop85 | combined_v3_augmented | 0.7270 | -0.0050 | 0.7997 | +0.0013 |
| crop85 | resnet18 | 0.8100 | -0.0060 | 0.8938 | -0.0029 |
| crop85 | resnet18_augmented | 0.7920 | -0.0150 | 0.8809 | -0.0070 |
| jpeg70 | combined_v3 | 0.7340 | +0.0020 | 0.7994 | -0.0033 |
| jpeg70 | combined_v3_augmented | 0.7420 | +0.0100 | 0.7983 | -0.0001 |
| jpeg70 | resnet18 | 0.8170 | +0.0010 | 0.8974 | +0.0007 |
| jpeg70 | resnet18_augmented | 0.8050 | -0.0020 | 0.8879 | -0.0001 |
| resize_half | combined_v3 | 0.6940 | -0.0380 | 0.7920 | -0.0107 |
| resize_half | combined_v3_augmented | 0.7230 | -0.0090 | 0.8036 | +0.0052 |
| resize_half | resnet18 | 0.8110 | -0.0050 | 0.8938 | -0.0030 |
| resize_half | resnet18_augmented | 0.8050 | -0.0020 | 0.8874 | -0.0005 |

## Interpretation

Neural train-time augmentation did what it was supposed to do for some transforms, but not enough to beat the unaugmented ResNet-18 overall. Under blur, the augmented ResNet-18 drops only 0.4 accuracy points from its own clean score, compared with a 1.7-point drop for the original ResNet-18. Under half-resolution resize, it drops 0.2 points, compared with 0.5 points for the original.

The tradeoff is clean accuracy and crop behavior. The original ResNet-18 remains the best absolute neural model on clean validation, JPEG, resize, and crop. The augmented ResNet-18 only edges the original under blur accuracy, 0.8030 versus 0.7990, while its AUC is still slightly lower.

The conventional baseline benefits more dramatically from the same augmentation recipe because its handcrafted signal features are more transform-sensitive. `combined_v3_augmented` improves blur accuracy from 0.6350 to 0.6980 and resize accuracy from 0.6940 to 0.7230, while keeping the same clean accuracy.

This suggests the next neural pass should use either lighter augmentation sampling, a longer schedule, threshold calibration, or a stronger backbone rather than cloning every transformed image for every epoch.

## New Dataset Leads

A fresh Kaggle search on 2026-06-12 found several recent candidates beyond the datasets already benchmarked here:

- `shivaligupta17/finetunedataset`: updated 2026-05-31, about 398 MB. The file listing shows `new dataset/train/fake/*.png`, making it the strongest immediate candidate if a matching real folder is present.
- `ishu15m/ai-vs-real-images`: updated 2026-05-28, about 603 MB. The listing shows AI category folders such as `AI-images/AI-images/ai_animals`, so it needs label mapping validation before direct use.
- `itszubi/ai-vs-real-images-classification-dataset`: updated 2026-04-27, about 84 MB. The listing begins with topic folders such as `animals/*.png`; inspect labels before using it as a binary benchmark.
- `josiahsafley/safemedia-ai-image-evaluation-sample`: updated 2026-05-09, about 73 MB. The listing contains `metadata.json` and `target.png` pairs, useful for generated-image qualitative probes but not a ready real-vs-fake folder.

These were added to `configs/datasets.json` as catalog candidates. The next import pass should download only small samples first and verify label folders before running full benchmarks.

## Reproduce

```powershell
python scripts/train_neural_net.py `
  --data-dir data/raw/ms_cocoai_2026_subset_500 `
  --output-dir runs/ms_cocoai_2026_subset_500/resnet18_augmented `
  --model resnet18 `
  --pretrained `
  --epochs 4 `
  --batch-size 64 `
  --image-size 128 `
  --num-workers 0 `
  --device cuda `
  --train-augment-variants jpeg70 blur1 resize_half crop85
```

```powershell
python scripts/evaluate_neural_net.py `
  --model-dir runs/ms_cocoai_2026_subset_500/resnet18_augmented `
  --target-dir data/raw/ms_cocoai_2026_validation_source_balanced_100 `
  --output-dir runs/ms_cocoai_2026_source_balanced_eval/resnet18_augmented `
  --image-size 128 `
  --batch-size 64 `
  --num-workers 0 `
  --device cuda `
  --target-split all
```

Repeat the evaluation command for each folder under `data/raw/ms_cocoai_2026_source_balanced_robustness_variants`, then summarize:

```powershell
python scripts/summarize_robustness_eval.py `
  --baseline combined_v3=runs/ms_cocoai_2026_source_balanced_eval/combined_v3/metrics.json `
  --baseline combined_v3_augmented=runs/ms_cocoai_2026_source_balanced_eval/combined_v3_augmented/metrics.json `
  --baseline resnet18=runs/ms_cocoai_2026_source_balanced_eval/resnet18/metrics.json `
  --baseline resnet18_augmented=runs/ms_cocoai_2026_source_balanced_eval/resnet18_augmented/metrics.json `
  --metrics blur1:combined_v3=runs/ms_cocoai_2026_robustness_eval/blur1/combined_v3/metrics.json `
  --metrics blur1:combined_v3_augmented=runs/ms_cocoai_2026_robustness_eval/blur1/combined_v3_augmented/metrics.json `
  --metrics blur1:resnet18=runs/ms_cocoai_2026_robustness_eval/blur1/resnet18/metrics.json `
  --metrics blur1:resnet18_augmented=runs/ms_cocoai_2026_robustness_eval/blur1/resnet18_augmented/metrics.json `
  --metrics crop85:combined_v3=runs/ms_cocoai_2026_robustness_eval/crop85/combined_v3/metrics.json `
  --metrics crop85:combined_v3_augmented=runs/ms_cocoai_2026_robustness_eval/crop85/combined_v3_augmented/metrics.json `
  --metrics crop85:resnet18=runs/ms_cocoai_2026_robustness_eval/crop85/resnet18/metrics.json `
  --metrics crop85:resnet18_augmented=runs/ms_cocoai_2026_robustness_eval/crop85/resnet18_augmented/metrics.json `
  --metrics jpeg70:combined_v3=runs/ms_cocoai_2026_robustness_eval/jpeg70/combined_v3/metrics.json `
  --metrics jpeg70:combined_v3_augmented=runs/ms_cocoai_2026_robustness_eval/jpeg70/combined_v3_augmented/metrics.json `
  --metrics jpeg70:resnet18=runs/ms_cocoai_2026_robustness_eval/jpeg70/resnet18/metrics.json `
  --metrics jpeg70:resnet18_augmented=runs/ms_cocoai_2026_robustness_eval/jpeg70/resnet18_augmented/metrics.json `
  --metrics resize_half:combined_v3=runs/ms_cocoai_2026_robustness_eval/resize_half/combined_v3/metrics.json `
  --metrics resize_half:combined_v3_augmented=runs/ms_cocoai_2026_robustness_eval/resize_half/combined_v3_augmented/metrics.json `
  --metrics resize_half:resnet18=runs/ms_cocoai_2026_robustness_eval/resize_half/resnet18/metrics.json `
  --metrics resize_half:resnet18_augmented=runs/ms_cocoai_2026_robustness_eval/resize_half/resnet18_augmented/metrics.json `
  --out-dir runs/ms_cocoai_2026_robustness_eval/summary_all_augmented
```
