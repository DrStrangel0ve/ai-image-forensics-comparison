# Defactify / MS COCOAI Augmented Conventional Robustness

Run date: 2026-06-12

This follow-up trains the strongest conventional stack, `combined_v3`, with deterministic train-time variants:

- `jpeg70`
- `blur1`
- `resize_half`
- `crop85`

The goal is to test whether a conventional forensic baseline can be made less brittle to common image-sharing transformations without changing the neural baseline. The transforms are now shared in `src/forensic_compare/transforms.py`, and `scripts/run_feature_baseline.py` can apply them in memory with `--train-augment-variants`.

Dataset context is unchanged from the source-balanced Defactify/MS COCOAI validation run: 500 real images and 500 generated images, balanced across SD2.1, SDXL, SD3, DALL-E 3, and Midjourney v6.

## Clean Validation

| method | accuracy | precision | recall | f1 | roc_auc | n_samples |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| combined_v3 | 0.7320 | 0.7248 | 0.7480 | 0.7362 | 0.8027 | 1000 |
| combined_v3_augmented | 0.7320 | 0.7417 | 0.7120 | 0.7265 | 0.7984 | 1000 |
| resnet18 | 0.8160 | 0.8098 | 0.8260 | 0.8178 | 0.8967 | 1000 |

The augmented conventional model keeps the same clean accuracy as the prior `combined_v3` model. Its precision is higher and recall is lower, so the augmentation shifts the decision behavior slightly toward fewer generated-image calls.

## Robustness Results

Deltas are relative to each method's clean source-balanced validation score.

| variant | method | accuracy | accuracy delta | roc_auc | roc_auc delta |
| --- | --- | ---: | ---: | ---: | ---: |
| blur1 | combined_v3 | 0.6350 | -0.0970 | 0.7665 | -0.0362 |
| blur1 | combined_v3_augmented | 0.6980 | -0.0340 | 0.7864 | -0.0119 |
| blur1 | resnet18 | 0.7990 | -0.0170 | 0.8879 | -0.0089 |
| crop85 | combined_v3 | 0.7140 | -0.0180 | 0.7977 | -0.0049 |
| crop85 | combined_v3_augmented | 0.7270 | -0.0050 | 0.7997 | +0.0013 |
| crop85 | resnet18 | 0.8100 | -0.0060 | 0.8938 | -0.0029 |
| jpeg70 | combined_v3 | 0.7340 | +0.0020 | 0.7994 | -0.0033 |
| jpeg70 | combined_v3_augmented | 0.7420 | +0.0100 | 0.7983 | -0.0001 |
| jpeg70 | resnet18 | 0.8170 | +0.0010 | 0.8974 | +0.0007 |
| resize_half | combined_v3 | 0.6940 | -0.0380 | 0.7920 | -0.0107 |
| resize_half | combined_v3_augmented | 0.7230 | -0.0090 | 0.8036 | +0.0052 |
| resize_half | resnet18 | 0.8110 | -0.0050 | 0.8938 | -0.0030 |

## Interpretation

Train-time robustness augmentation materially improves the conventional baseline under transformations. The biggest recovery is blur: `combined_v3` dropped 9.7 accuracy points, while `combined_v3_augmented` drops 3.4 points. Half-resolution resize also improves, shrinking the loss from 3.8 points to 0.9 points.

The neural baseline still has a large absolute lead: ResNet-18 is 10.1 points ahead of the augmented conventional model under blur and 8.8 points ahead under resize. So augmentation closes the brittleness gap but does not close the representation gap.

This result is also cleaner than threshold-only calibration. The earlier calibration report recovered blurred `combined_v3` from 0.6350 to 0.6740 using a clean threshold, while train-time augmentation reaches 0.6980 at the default threshold and improves AUC as well.

## New Dataset Leads

This pass also validated newer Hugging Face candidates through Dataset Viewer:

- OpenFake: `ComplexDataLab/OpenFake`, 2,493,222 rows with core train/validation/test and Reddit test splits. It includes labels, generator/model metadata, content type, and release dates. https://huggingface.co/datasets/ComplexDataLab/OpenFake
- Parveshiiii/AI-vs-Real: 13,999 rows in a single train split, practical enough for a direct import pass. https://huggingface.co/datasets/Parveshiiii/AI-vs-Real
- Real vs AI Corpus: `Zitacron/real-vs-ai-corpus`, 12,336,207 rows with source dataset/license metadata. It is a large streaming/sample target rather than a local full download. https://huggingface.co/datasets/Zitacron/real-vs-ai-corpus
- GPT-image-2 public outputs: `Scam-AI/gpt-image-2` is a recent generated-only candidate, but Dataset Viewer returned 401 Unauthorized on this run, so it remains a manual/gated import target. https://huggingface.co/datasets/Scam-AI/gpt-image-2

## Reproduce

```powershell
python scripts/run_feature_baseline.py `
  --data-dir data/raw/ms_cocoai_2026_subset_500 `
  --output-dir runs/ms_cocoai_2026_subset_500/feature_combined_v3_augmented_logistic_regression `
  --feature-set combined_v3 `
  --classifier logistic_regression `
  --image-size 128 `
  --train-augment-variants jpeg70 blur1 resize_half crop85 `
  --skip-errors
```

```powershell
python scripts/evaluate_feature_model.py `
  --model-dir runs/ms_cocoai_2026_subset_500/feature_combined_v3_augmented_logistic_regression `
  --target-dir data/raw/ms_cocoai_2026_validation_source_balanced_100 `
  --output-dir runs/ms_cocoai_2026_source_balanced_eval/combined_v3_augmented `
  --image-size 128 `
  --target-split all `
  --skip-errors
```

Repeat the evaluation command for each folder under `data/raw/ms_cocoai_2026_source_balanced_robustness_variants`, then run:

```powershell
python scripts/summarize_robustness_eval.py `
  --out-dir runs/ms_cocoai_2026_robustness_eval/summary_augmented `
  --baseline combined_v3=runs/ms_cocoai_2026_source_balanced_eval/combined_v3/metrics.json `
  --baseline combined_v3_augmented=runs/ms_cocoai_2026_source_balanced_eval/combined_v3_augmented/metrics.json `
  --baseline resnet18=runs/ms_cocoai_2026_source_balanced_eval/resnet18/metrics.json `
  --metrics blur1:combined_v3=runs/ms_cocoai_2026_robustness_eval/blur1/combined_v3/metrics.json `
  --metrics blur1:combined_v3_augmented=runs/ms_cocoai_2026_robustness_eval/blur1/combined_v3_augmented/metrics.json `
  --metrics blur1:resnet18=runs/ms_cocoai_2026_robustness_eval/blur1/resnet18/metrics.json `
  --metrics crop85:combined_v3=runs/ms_cocoai_2026_robustness_eval/crop85/combined_v3/metrics.json `
  --metrics crop85:combined_v3_augmented=runs/ms_cocoai_2026_robustness_eval/crop85/combined_v3_augmented/metrics.json `
  --metrics crop85:resnet18=runs/ms_cocoai_2026_robustness_eval/crop85/resnet18/metrics.json `
  --metrics jpeg70:combined_v3=runs/ms_cocoai_2026_robustness_eval/jpeg70/combined_v3/metrics.json `
  --metrics jpeg70:combined_v3_augmented=runs/ms_cocoai_2026_robustness_eval/jpeg70/combined_v3_augmented/metrics.json `
  --metrics jpeg70:resnet18=runs/ms_cocoai_2026_robustness_eval/jpeg70/resnet18/metrics.json `
  --metrics resize_half:combined_v3=runs/ms_cocoai_2026_robustness_eval/resize_half/combined_v3/metrics.json `
  --metrics resize_half:combined_v3_augmented=runs/ms_cocoai_2026_robustness_eval/resize_half/combined_v3_augmented/metrics.json `
  --metrics resize_half:resnet18=runs/ms_cocoai_2026_robustness_eval/resize_half/resnet18/metrics.json
```
