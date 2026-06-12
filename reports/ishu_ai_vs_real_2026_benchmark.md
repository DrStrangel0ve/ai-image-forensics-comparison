# Ishu AI vs Real 2026 Benchmark

Run date: 2026-06-12

This pass validates the recent Kaggle dataset `ishu15m/ai-vs-real-images`, listed as "ai-vs-real-images" and updated on 2026-05-28.

Dataset URL: https://www.kaggle.com/datasets/ishu15m/ai-vs-real-images

## Dataset Audit

The download exposes a single-folder binary layout that the project can use directly:

```text
ishu_ai_vs_real_2026/
  AI-images/
    AI-images/
      ai_animals/
      ai_buildings/
      ai_food/
      ai_human/
      ai_interior/
      ai_items/
      ai_nature/
  Real-images/
    Real-images/
      real_animals/
      real_buildings/
      real_food/
      real_humans/
      real_interior/
      real_items/
      real_nature/
```

Because there is no fixed train/test split, the benchmark scripts use their deterministic stratified split: 453 train images and 114 test images.

| class | n_images |
| --- | ---: |
| AI-images | 278 |
| Real-images | 289 |

Category counts:

| category | AI | real |
| --- | ---: | ---: |
| animals | 50 | 50 |
| buildings | 50 | 53 |
| food | 31 | 30 |
| human | 35 | 50 |
| interior | 33 | 22 |
| items | 34 | 34 |
| nature | 45 | 50 |

Audit summary from `scripts/audit_image_dataset.py`:

| field | value |
| --- | ---: |
| images | 567 |
| unique SHA-256 hashes | 566 |
| exact duplicate groups | 1 |
| perceptual near-duplicate pairs | 4 |
| cross-class duplicate groups | 0 |
| cross-class near-duplicate pairs | 0 |
| width range | 165..8192 |
| height range | 159..6240 |

The exact and perceptual duplicate candidates are all within `AI-images/ai_food`. Because this dataset has a single-folder layout, there is no upstream fixed split to audit for cross-split leakage; the deterministic split is created by path hashing at training time.

## In-Dataset Benchmark

| method | accuracy | precision | recall | f1 | roc_auc | n_samples |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| photometric | 0.7105 | 0.6769 | 0.7857 | 0.7273 | 0.7509 | 114 |
| noise | 0.6667 | 0.6452 | 0.7143 | 0.6780 | 0.6915 | 114 |
| combined_v3 | 0.8158 | 0.8070 | 0.8214 | 0.8142 | 0.8938 | 114 |
| resnet18 | 0.7719 | 0.7679 | 0.7679 | 0.7679 | 0.8608 | 114 |

This is a useful counterexample to the Defactify/MS COCOAI pattern: the strongest conventional feature stack beats the six-epoch pretrained ResNet-18 on both accuracy and AUC.

## Category Breakdown

Accuracy by nested category on the deterministic test split:

| category | n_test | fake_test | combined_v3_acc | resnet18_acc |
| --- | ---: | ---: | ---: | ---: |
| animals | 20 | 11 | 0.8000 | 0.6500 |
| buildings | 18 | 11 | 0.9444 | 0.8889 |
| food | 6 | 3 | 0.8333 | 1.0000 |
| human | 22 | 6 | 0.8182 | 0.7727 |
| interior | 16 | 8 | 0.8750 | 0.7500 |
| items | 14 | 6 | 0.8571 | 0.9286 |
| nature | 18 | 11 | 0.6111 | 0.6111 |

`nature` is the hardest category for both methods. The conventional stack does especially well on buildings, interiors, and human images, while ResNet-18 is strongest on food and items in this split.

## MS COCOAI Zero-Shot Transfer

The same two model families trained on Defactify/MS COCOAI were also evaluated against all 567 Ishu images.

| source model | accuracy | precision | recall | f1 | roc_auc | n_samples |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| combined_v3 | 0.5608 | 0.5443 | 0.6403 | 0.5884 | 0.5734 | 567 |
| resnet18 | 0.6243 | 0.5905 | 0.7626 | 0.6656 | 0.7003 | 567 |

The transfer ranking reverses the in-dataset result. `combined_v3` wins after fitting directly on the Ishu split, but the MS COCOAI ResNet-18 transfers better than the MS COCOAI `combined_v3` model. That is a useful warning: the conventional cues are strong but dataset-specific, and the neural model still has the better cross-domain representation here.

## Interpretation

This dataset is now a validated, practical-size May 2026 benchmark with more image variety than the tiny ChatGPT/Gemini probe. It is still small enough that one deterministic split should not be treated as final; a future pass should add repeated split evaluation or k-fold cross-validation.

The photometric normal-consistency baseline is meaningfully above chance at 0.7509 AUC, and the full `combined_v3` conventional stack is the best in-dataset model. The result supports keeping conventional signal features in the comparison rather than treating them as weak baselines.

The zero-shot transfer result points in the opposite direction: training-domain coverage still matters more than raw in-dataset accuracy. The next best robustness step is to run repeated splits and then test transform robustness on this Ishu dataset, especially for the weak `nature` category.

## Reproduce

```powershell
python scripts/download_dataset.py ishu_ai_vs_real_2026
```

```powershell
python scripts/audit_image_dataset.py `
  --data-dir data/raw/ishu_ai_vs_real_2026 `
  --out-dir runs/ishu_ai_vs_real_2026_initial/audit
```

```powershell
python scripts/run_benchmark.py `
  --data-dir data/raw/ishu_ai_vs_real_2026 `
  --out-dir runs/ishu_ai_vs_real_2026_initial `
  --methods photometric noise combined_v3 neural `
  --feature-classifier logistic_regression `
  --feature-image-size 128 `
  --neural-model resnet18 `
  --pretrained `
  --epochs 6 `
  --batch-size 64 `
  --neural-image-size 128 `
  --num-workers 0 `
  --device cuda `
  --skip-errors
```

```powershell
python scripts/evaluate_feature_model.py `
  --model-dir runs/ms_cocoai_2026_subset_500/feature_combined_v3_logistic_regression `
  --target-dir data/raw/ishu_ai_vs_real_2026 `
  --output-dir runs/ishu_ai_vs_real_2026_cross_ms_cocoai/combined_v3 `
  --image-size 128 `
  --target-split all `
  --skip-errors
```

```powershell
python scripts/evaluate_neural_net.py `
  --model-dir runs/ms_cocoai_2026_subset_500/resnet18 `
  --target-dir data/raw/ishu_ai_vs_real_2026 `
  --output-dir runs/ishu_ai_vs_real_2026_cross_ms_cocoai/resnet18 `
  --image-size 128 `
  --batch-size 64 `
  --num-workers 0 `
  --device cuda `
  --target-split all
```
