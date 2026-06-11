# CIFAKE Subset Initial Result

Date: 2026-06-12

Dataset:

```text
birdy654/cifake-real-and-ai-generated-synthetic-images
```

Run scope:

- Train subset: 2,000 balanced images.
- Test subset: 1,000 balanced images.
- Neural baseline: ResNet-18, one epoch, image size 96.
- Photometric baseline: single-image normal-consistency features, logistic regression, image size 128.
- Device used for this run: CPU. The local Python environment had a CPU-only PyTorch wheel at run time; install `requirements-gpu-cu128.txt` to use the RTX 3060 Ti.

Commands:

```powershell
python scripts/run_photometric_baseline.py `
  --data-dir data/raw/cifake `
  --output-dir runs/cifake_subset/photometric `
  --image-size 128 `
  --max-train-samples 2000 `
  --max-test-samples 1000

python scripts/train_neural_net.py `
  --data-dir data/raw/cifake `
  --output-dir runs/cifake_subset/resnet18 `
  --model resnet18 `
  --epochs 1 `
  --batch-size 64 `
  --image-size 96 `
  --num-workers 0 `
  --max-train-samples 2000 `
  --max-test-samples 1000 `
  --device auto

python scripts/compare_methods.py `
  --neural runs/cifake_subset/resnet18/metrics.json `
  --photometric runs/cifake_subset/photometric/metrics.json `
  --out-dir runs/cifake_subset/comparison
```

Results:

| method | accuracy | precision | recall | f1 | roc_auc | n_samples |
| --- | --- | --- | --- | --- | --- | --- |
| neural_net | 0.7450 | 0.7207 | 0.8000 | 0.7583 | 0.8532 | 1000 |
| photometric_normal_consistency | 0.8120 | 0.8391 | 0.7720 | 0.8042 | 0.9091 | 1000 |

On this small subset, the photometric normal-consistency baseline outperformed the one-epoch ResNet-18 run. This should be treated as an initial sanity benchmark, not the final conclusion. The neural baseline should improve with CUDA-enabled PyTorch, more epochs, and a full-dataset run.
