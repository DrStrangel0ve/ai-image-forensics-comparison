# AI vs Real 2026 Benchmark

Date: 2026-06-12

Dataset:

```text
muqaddasejaz/ai-generated-vs-real-images-dataset
```

Local layout:

```text
data/raw/ai_vs_real_2026/Dataset/
  AI Images/AI/
  Real Images/Real/
```

Run scope:

- Usable image-folder examples: 973 images after aligning feature extraction to torchvision-supported image extensions.
- Split: stratified single-folder split with 778 train images and 195 test images.
- Neural baseline: ImageNet-pretrained ResNet-18, 5 epochs, image size 128.
- Conventional baselines: logistic regression over photometric, noise/compression/frequency, and combined features.
- Device: CUDA on NVIDIA GeForce RTX 3060 Ti.
- Best neural checkpoint selection: epoch 4.

Commands:

```powershell
.\.venv\Scripts\python scripts\run_benchmark.py `
  --dataset-key ai_vs_real_2026 `
  --out-dir runs\ai_vs_real_2026_full `
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
| feature_photometric | 0.6513 | 0.6754 | 0.7130 | 0.6937 | 0.6907 | 195 |
| feature_noise | 0.6359 | 0.6832 | 0.6389 | 0.6603 | 0.6674 | 195 |
| feature_combined | 0.6821 | 0.7212 | 0.6944 | 0.7075 | 0.7362 | 195 |
| neural_resnet18 | 0.8205 | 0.8288 | 0.8519 | 0.8402 | 0.8992 | 195 |

Takeaways:

- Pretrained ResNet-18 won this run, especially by ROC AUC.
- Combining photometric and noise/compression features beat either conventional feature family alone.
- The conventional baselines remain useful because they are cheaper, more interpretable, and expose physical/signal artifacts that can be tracked across datasets.
- This is still a small dataset; the next stronger result should use a larger 2026 dataset or cross-dataset testing.
