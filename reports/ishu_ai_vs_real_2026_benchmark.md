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

On this seed-7 split, the strongest conventional feature stack beats the six-epoch pretrained ResNet-18 on both accuracy and AUC. Because the dataset is still modest in size, the result is best treated as a single-split finding rather than a stable ranking by itself.

## Repeated Split Check

Additional deterministic splits with `--seed 17` and `--seed 29` reran the two strongest in-dataset methods using the same 80/20 validation fraction and model settings.

| seed | method | accuracy | precision | recall | f1 | roc_auc |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 7 | combined_v3 | 0.8158 | 0.8070 | 0.8214 | 0.8142 | 0.8938 |
| 7 | resnet18 | 0.7719 | 0.7679 | 0.7679 | 0.7679 | 0.8608 |
| 17 | combined_v3 | 0.8246 | 0.8000 | 0.8571 | 0.8276 | 0.9089 |
| 17 | resnet18 | 0.8509 | 0.8679 | 0.8214 | 0.8440 | 0.8981 |
| 29 | combined_v3 | 0.8333 | 0.8136 | 0.8571 | 0.8348 | 0.8799 |
| 29 | resnet18 | 0.8509 | 0.8305 | 0.8750 | 0.8522 | 0.9190 |

Three-run summary:

| method | n_runs | accuracy_mean | accuracy_std | roc_auc_mean | roc_auc_std | accuracy_wins | auc_wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| combined_v3 | 3 | 0.8246 | 0.0072 | 0.8942 | 0.0118 | 1 | 2 |
| resnet18 | 3 | 0.8246 | 0.0372 | 0.8927 | 0.0241 | 2 | 1 |

This overturns the original single-split conclusion. Across three seeds, `combined_v3` and ResNet-18 have identical mean accuracy to four decimals and nearly identical mean AUC. ResNet-18 wins accuracy on two splits and AUC on seed 29, while `combined_v3` has lower variance and wins AUC on two splits. More repeated seeds or k-fold validation are needed before claiming either family is the stable in-dataset winner.

## Physics-Guided Neural Fusion

Because the repeated split check made the conventional `combined_v3` stack and ResNet-18 look nearly tied, this follow-up adds a physics-guided neural fusion baseline. This is not a calibrated multi-light photometric-stereo PINN; the available datasets are single-image real-vs-generated corpora. Instead, the model fuses a pretrained ResNet-18 image embedding with a small MLP over standardized `combined_v3` forensic features: photometric normal-consistency, noise residuals, JPEG recompression response, 8x8 residual periodicity, residual RGB correlation, local residual variance, and chroma statistics.

Per-seed comparison:

| seed | method | accuracy | precision | recall | f1 | roc_auc |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 7 | combined_v3 | 0.8158 | 0.8070 | 0.8214 | 0.8142 | 0.8938 |
| 7 | resnet18 | 0.7719 | 0.7679 | 0.7679 | 0.7679 | 0.8608 |
| 7 | physics_guided_resnet18_combined_v3 | 0.7982 | 0.7705 | 0.8393 | 0.8034 | 0.8808 |
| 17 | combined_v3 | 0.8246 | 0.8000 | 0.8571 | 0.8276 | 0.9089 |
| 17 | resnet18 | 0.8509 | 0.8679 | 0.8214 | 0.8440 | 0.8981 |
| 17 | physics_guided_resnet18_combined_v3 | 0.8596 | 0.8704 | 0.8393 | 0.8545 | 0.9372 |
| 29 | combined_v3 | 0.8333 | 0.8136 | 0.8571 | 0.8348 | 0.8799 |
| 29 | resnet18 | 0.8509 | 0.8305 | 0.8750 | 0.8522 | 0.9190 |
| 29 | physics_guided_resnet18_combined_v3 | 0.8772 | 0.8500 | 0.9107 | 0.8793 | 0.9350 |

Three-run summary:

| method | n_runs | accuracy_mean | accuracy_std | roc_auc_mean | roc_auc_std | accuracy_wins | auc_wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| combined_v3 | 3 | 0.8246 | 0.0072 | 0.8942 | 0.0118 | 1 | 1 |
| resnet18 | 3 | 0.8246 | 0.0372 | 0.8927 | 0.0241 | 0 | 0 |
| physics_guided_resnet18_combined_v3 | 3 | 0.8450 | 0.0338 | 0.9177 | 0.0261 | 2 | 2 |

The fusion model is the first in-dataset method here to clearly beat both the conventional stack and vanilla ResNet-18 on the three-seed mean. It still underperforms `combined_v3` on seed 7, so the result should be treated as promising rather than settled. Follow-up reports now cover Ishu seed-29 transfer to MS COCOAI and robustness transforms for the fused model.

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

The same three model families trained on Defactify/MS COCOAI were also evaluated against all 567 Ishu images.

| source model | accuracy | precision | recall | f1 | roc_auc | n_samples |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| combined_v3 | 0.5608 | 0.5443 | 0.6403 | 0.5884 | 0.5734 | 567 |
| resnet18 | 0.6243 | 0.5905 | 0.7626 | 0.6656 | 0.7003 | 567 |
| physics_guided_resnet18_combined_v3 | 0.5873 | 0.5531 | 0.8237 | 0.6618 | 0.7089 | 567 |

The reverse transfer ranking is more nuanced than the in-dataset repeated-split result. The MS COCOAI ResNet-18 has the best default-threshold accuracy, but the MS COCOAI-trained fusion model has the best AUC. Fusion gains +0.0086 AUC over ResNet-18 and +0.1355 AUC over `combined_v3`, but loses 3.7 accuracy points to ResNet-18 because it over-flags real Ishu images as generated at the default 0.5 threshold. A source-domain threshold-calibration follow-up in `reports/cross_dataset_2026_generalization.md` fixes most of that: using the MS COCOAI source threshold raises fusion transfer accuracy from 0.5873 to 0.6596.

## Interpretation

This dataset is now a validated, practical-size May 2026 benchmark with more image variety than the tiny ChatGPT/Gemini probe. It is still small enough that three deterministic splits should not be treated as final; a future pass should broaden the repeated-seed sweep or add k-fold cross-validation.

The photometric normal-consistency baseline is meaningfully above chance at 0.7509 AUC, and the full `combined_v3` conventional stack remains competitive with the six-epoch pretrained ResNet-18 in the repeated split check. The physics-guided fusion model is currently strongest in-dataset, which supports keeping conventional signal features in the comparison and feeding them into neural models rather than treating them as weak standalone baselines.

The zero-shot transfer result points in the opposite direction from the Ishu in-dataset result: training-domain coverage and score calibration still matter more than raw same-dataset accuracy. Ishu seed-29 to MS COCOAI found the fused model ahead of both unfused branches at 0.6330 accuracy / 0.6923 AUC. MS COCOAI back to Ishu found fusion ahead by AUC, and source-domain threshold calibration makes it the best transfer-accuracy model too. The next best robustness step is to repeat cross-domain transfer across more seeds and reserve a clean source calibration split, especially for the weak `nature` category.

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
  --seed 7 `
  --val-fraction 0.2 `
  --skip-errors
```

```powershell
python scripts/run_benchmark.py `
  --data-dir data/raw/ishu_ai_vs_real_2026 `
  --out-dir runs/ishu_ai_vs_real_2026_seed17 `
  --methods combined_v3 neural `
  --feature-classifier logistic_regression `
  --feature-image-size 128 `
  --neural-model resnet18 `
  --pretrained `
  --epochs 6 `
  --batch-size 64 `
  --neural-image-size 128 `
  --num-workers 0 `
  --device cuda `
  --seed 17 `
  --val-fraction 0.2 `
  --skip-errors
```

The seed sweep can now be automated:

```powershell
python scripts/run_repeated_benchmark.py `
  --data-dir data/raw/ishu_ai_vs_real_2026 `
  --out-dir runs/ishu_ai_vs_real_2026_physics_guided_repeated_auto `
  --seeds 7 17 29 `
  -- `
  --methods combined_v3 neural physics_guided `
  --feature-classifier logistic_regression `
  --feature-image-size 128 `
  --neural-model resnet18 `
  --pretrained `
  --epochs 6 `
  --batch-size 64 `
  --neural-image-size 128 `
  --num-workers 0 `
  --device cuda `
  --val-fraction 0.2 `
  --physics-feature-set combined_v3 `
  --physics-feature-image-size 128 `
  --skip-errors
```

Single physics-guided run:

```powershell
python scripts/run_benchmark.py `
  --data-dir data/raw/ishu_ai_vs_real_2026 `
  --out-dir runs/ishu_ai_vs_real_2026_physics_guided_seed29 `
  --methods physics_guided `
  --feature-classifier logistic_regression `
  --feature-image-size 128 `
  --neural-model resnet18 `
  --pretrained `
  --epochs 6 `
  --batch-size 64 `
  --neural-image-size 128 `
  --num-workers 0 `
  --device cuda `
  --seed 29 `
  --val-fraction 0.2 `
  --physics-feature-set combined_v3 `
  --physics-feature-image-size 128 `
  --skip-errors
```

Evaluate the saved physics-guided model on the same deterministic seed-29 test split:

```powershell
python scripts/evaluate_physics_guided_net.py `
  --model-dir runs/ishu_ai_vs_real_2026_physics_guided_seed29/physics_guided_resnet18_combined_v3 `
  --target-dir data/raw/ishu_ai_vs_real_2026 `
  --output-dir runs/ishu_ai_vs_real_2026_physics_guided_seed29/eval_repro_test `
  --image-size 128 `
  --feature-image-size 128 `
  --batch-size 64 `
  --num-workers 0 `
  --device cuda `
  --target-split test `
  --seed 29 `
  --val-fraction 0.2 `
  --skip-errors
```

```powershell
python scripts/compare_methods.py `
  --out-dir runs/ishu_ai_vs_real_2026_repeated_splits/seed7_comparison `
  --metrics feature_combined_v3=runs/ishu_ai_vs_real_2026_initial/feature_combined_v3_logistic_regression/metrics.json `
  --metrics neural_resnet18=runs/ishu_ai_vs_real_2026_initial/resnet18/metrics.json
```

```powershell
python scripts\summarize_repeated_benchmarks.py `
  --comparison seed7=runs/ishu_ai_vs_real_2026_repeated_splits/seed7_comparison/comparison.csv `
  --comparison seed17=runs/ishu_ai_vs_real_2026_seed17/comparison/comparison.csv `
  --comparison seed29=runs/ishu_ai_vs_real_2026_repeated_splits_auto/seed29/comparison/comparison.csv `
  --out-dir runs/ishu_ai_vs_real_2026_repeated_splits/summary_3seed
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

python scripts/evaluate_physics_guided_net.py `
  --model-dir runs/ms_cocoai_2026_subset_500/physics_guided_resnet18_combined_v3 `
  --target-dir data/raw/ishu_ai_vs_real_2026 `
  --output-dir runs/ishu_ai_vs_real_2026_cross_ms_cocoai/physics_guided_resnet18_combined_v3 `
  --image-size 128 `
  --feature-image-size 128 `
  --batch-size 64 `
  --num-workers 0 `
  --device cuda `
  --target-split all `
  --skip-errors
```
