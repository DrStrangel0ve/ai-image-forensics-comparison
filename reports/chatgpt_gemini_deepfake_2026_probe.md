# ChatGPT/Gemini Deepfake 2026 Probe

Run date: 2026-06-12

This pass validates the recent Kaggle dataset `shivaligupta17/finetunedataset`, listed as "Chatgpt/Gemini Realistic images for DeepfakeDetect" and updated on 2026-05-31.

Dataset URL: https://www.kaggle.com/datasets/shivaligupta17/finetunedataset

## Dataset Audit

The download extracts directly into a usable image-folder layout:

```text
new dataset/
  train/
    fake/
    real/
  valid/
    fake/
    real/
```

The new `scripts/audit_image_dataset.py` command checks class counts, image dimensions, and exact SHA-256 duplicates.

| split | fake | real |
| --- | ---: | ---: |
| train | 100 | 102 |
| valid | 21 | 20 |

Audit summary:

| field | value |
| --- | ---: |
| images | 243 |
| unique SHA-256 hashes | 238 |
| exact duplicate groups | 5 |
| cross-split duplicate groups | 0 |
| cross-class duplicate groups | 0 |
| width range | 928..1537 |
| height range | 768..1448 |

The exact duplicate pairs are all within `train/fake`, mostly duplicate filenames with `(1)` suffixes. There is no exact train/validation leakage.

## In-Dataset Benchmark

The first benchmark trains on the dataset's `train` split and evaluates on its 41-image `valid` split.

| method | accuracy | precision | recall | f1 | roc_auc | n_samples |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| photometric | 0.8537 | 0.8261 | 0.9048 | 0.8636 | 0.9048 | 41 |
| noise | 0.6585 | 0.6667 | 0.6667 | 0.6667 | 0.8429 | 41 |
| combined_v3 | 0.9268 | 0.9091 | 0.9524 | 0.9302 | 0.9714 | 41 |
| resnet18 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 41 |

This is a useful smoke benchmark, but it is too small to treat as a stable accuracy estimate. A single validation error moves accuracy by about 2.4 points.

## MS COCOAI Zero-Shot Transfer

The second check evaluates models trained on the Defactify/MS COCOAI 1,000-image subset against all 243 images in this ChatGPT/Gemini dataset.

| source model | accuracy | precision | recall | f1 | roc_auc | n_samples |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| combined_v3 | 0.5514 | 0.5316 | 0.8347 | 0.6495 | 0.5971 | 243 |
| combined_v3_augmented | 0.5185 | 0.5125 | 0.6777 | 0.5836 | 0.5484 | 243 |
| resnet18 | 0.4774 | 0.4867 | 0.9091 | 0.6340 | 0.5295 | 243 |
| resnet18_augmented | 0.3992 | 0.4309 | 0.6446 | 0.5166 | 0.3506 | 243 |

The transfer result is the real finding. The in-dataset split is easy, but MS COCOAI-trained detectors do not generalize cleanly to this small ChatGPT/Gemini domain. The unaugmented `combined_v3` feature model is best in this zero-shot check, but only at 0.5971 AUC.

## Interpretation

This dataset is now validated as a small, recent generated-image probe. It is valuable because it contains explicit ChatGPT/Gemini-style images from May 2026, but its validation set is tiny and should not drive strong claims by itself.

The photometric baseline does better here than on several harder benchmarks, reaching 0.8537 accuracy in-dataset. That means the generated images in this sample may contain visible shading, texture, or capture-pipeline inconsistencies that conventional features can exploit.

The poor MS COCOAI transfer result argues for the next heartbeat to import a larger recent dataset, preferably one with more generators and metadata. It also suggests adding threshold calibration and source-specific analysis to cross-dataset evaluations, because the current MS COCOAI models heavily over-call the ChatGPT/Gemini images as fake.

## Reproduce

```powershell
python scripts/download_dataset.py chatgpt_gemini_deepfake_2026
```

```powershell
python scripts/audit_image_dataset.py `
  --data-dir data/raw/chatgpt_gemini_deepfake_2026 `
  --out-dir runs/chatgpt_gemini_deepfake_2026_initial/audit
```

```powershell
python scripts/run_benchmark.py `
  --data-dir data/raw/chatgpt_gemini_deepfake_2026 `
  --out-dir runs/chatgpt_gemini_deepfake_2026_initial `
  --methods photometric noise combined_v3 neural `
  --feature-classifier logistic_regression `
  --feature-image-size 128 `
  --neural-model resnet18 `
  --pretrained `
  --epochs 8 `
  --batch-size 32 `
  --neural-image-size 128 `
  --num-workers 0 `
  --device cuda `
  --skip-errors
```

```powershell
python scripts/evaluate_feature_model.py `
  --model-dir runs/ms_cocoai_2026_subset_500/feature_combined_v3_logistic_regression `
  --target-dir data/raw/chatgpt_gemini_deepfake_2026 `
  --output-dir runs/chatgpt_gemini_deepfake_2026_cross_ms_cocoai/combined_v3 `
  --image-size 128 `
  --target-split all `
  --skip-errors
```

Repeat the transfer command for `combined_v3_augmented`, `resnet18`, and `resnet18_augmented`, then compare with `scripts/compare_methods.py`.
