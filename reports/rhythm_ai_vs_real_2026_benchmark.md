# Rhythm AI vs Real 2026 Benchmark

Date: 2026-06-12

Dataset:

```text
rhythmghai/ai-vs-real-images-dataset
```

Local layout:

```text
data/raw/rhythm_ai_vs_real_2026/
  Ai_generated_dataset/
    animals/
    city/
    food/
    nature/
    people/
  real_dataset/
    animals/
    city/
    food/
    nature/
    people/
```

Run scope:

- Usable image examples: 995 images.
- Split: stable stratified single-folder split with 796 train images and 199 test images.
- Neural baseline: ImageNet-pretrained ResNet-18, 5 epochs, image size 128.
- Conventional baselines: logistic regression over photometric, noise/compression/frequency, and combined features.
- Device: CUDA on NVIDIA GeForce RTX 3060 Ti.
- Best neural checkpoint selection: epoch 3.

Commands:

```powershell
.\.venv\Scripts\python scripts\download_dataset.py rhythm_ai_vs_real_2026

.\.venv\Scripts\python scripts\run_benchmark.py `
  --dataset-key rhythm_ai_vs_real_2026 `
  --out-dir runs\rhythm_ai_vs_real_2026_full `
  --methods photometric noise combined neural `
  --feature-classifier logistic_regression `
  --feature-image-size 128 `
  --neural-model resnet18 `
  --pretrained `
  --epochs 5 `
  --batch-size 64 `
  --neural-image-size 128 `
  --num-workers 0 `
  --device cuda
```

Results:

| method | accuracy | precision | recall | f1 | roc_auc | n_samples |
| --- | --- | --- | --- | --- | --- | --- |
| feature_photometric | 0.8693 | 0.7069 | 0.8200 | 0.7593 | 0.9420 | 199 |
| feature_noise | 0.8141 | 0.5942 | 0.8200 | 0.6891 | 0.9093 | 199 |
| feature_combined | 0.8693 | 0.6935 | 0.8600 | 0.7679 | 0.9522 | 199 |
| neural_resnet18 | 0.9698 | 0.9074 | 0.9800 | 0.9423 | 0.9956 | 199 |

Takeaways:

- ResNet-18 strongly won this dataset, with only 6 mistakes on the 199-image test split.
- Photometric and combined conventional features were much stronger here than on `ai_vs_real_2026`, suggesting this dataset exposes cleaner visual/signal artifacts.
- The combined conventional baseline had the best conventional F1 and ROC AUC, while photometric tied its accuracy.
- This result makes a cross-dataset generalization test the next useful step: train on one 2026 dataset, evaluate on the other without retraining.
