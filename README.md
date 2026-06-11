# AI Image Forensics Comparison

This project compares two ways to detect generated images:

1. A standard neural network classifier, defaulting to ResNet-18.
2. A photometric normal-consistency baseline that estimates pseudo normals from each image and checks whether lighting and surface gradients look physically coherent.
3. Conventional single-image forensic baselines that measure noise residuals, JPEG/block artifacts, ELA differences, FFT frequency balance, and chroma/noise consistency.

The default dataset target is Kaggle's **CIFAKE: Real and AI-Generated Synthetic Images**:

```text
birdy654/cifake-real-and-ai-generated-synthetic-images
```

That dataset has real CIFAR-style images and Stable Diffusion generated images with train/test folders, which makes it a clean first benchmark. The code also supports any image-folder dataset with class names such as `REAL` and `FAKE`.

## Important Photometric Note

True photometric stereo needs multiple images of the same object from the same camera under different known lighting directions. Most real-vs-generated Kaggle datasets are single-image datasets, including CIFAKE.

So this repo implements a **single-image photometric proxy**:

- estimate local normals from luminance gradients,
- measure normal-field integrability,
- measure high-frequency shading artifacts,
- measure edge, saturation, and color-channel consistency,
- train a small logistic-regression classifier from those physical features.

That gives a fair, runnable physics-inspired baseline against the neural network. If you later add a calibrated multi-light dataset, the photometric module is the place to extend it into full photometric stereo.

## Setup

From this repo root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For an NVIDIA desktop such as an RTX 3060 Ti, install the CUDA PyTorch wheels after the normal requirements:

```powershell
python -m pip install --force-reinstall -r requirements-gpu-cu128.txt
@'
import torch
print(torch.__version__, torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
'@ | python -
```

## Kaggle Access

The Kaggle API expects credentials at:

```text
C:\Users\<you>\.kaggle\kaggle.json
```

Download CIFAKE:

```powershell
python scripts/download_kaggle.py `
  --dataset birdy654/cifake-real-and-ai-generated-synthetic-images `
  --out data/raw/cifake
```

## Dataset Catalog

List known dataset candidates:

```powershell
python scripts/list_datasets.py
```

Download one by key:

```powershell
python scripts/download_dataset.py ai_vs_real_2026
```

The catalog currently includes:

- `cifake`: CIFAKE real vs Stable Diffusion images.
- `ai_vs_real_2026`: practical-size 2026 Kaggle real-vs-AI image dataset.
- `rhythm_ai_vs_real_2026`: another practical-size 2026 Kaggle real-vs-AI dataset.
- `stylegan3_faces_2026`: recent StyleGAN3 real-vs-fake face dataset.
- `ai_generated_vs_real_multigen`: large multi-generator Kaggle dataset.
- `ms_cocoai_2026`: Hugging Face research dataset with SD3, SD2.1, SDXL, DALL-E 3, and MidJourney v6 references.
- `realhd_2026`: external 2026 benchmark candidate.

## Run One Full Benchmark

The benchmark runner can compare multiple methods on the same split:

```powershell
python scripts/run_benchmark.py `
  --dataset-key ai_vs_real_2026 `
  --out-dir runs/ai_vs_real_2026_full `
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

## Run CIFAKE Manually

Train the neural network:

```powershell
python scripts/train_neural_net.py `
  --data-dir data/raw/cifake `
  --output-dir runs/resnet18 `
  --model resnet18 `
  --epochs 5 `
  --batch-size 128 `
  --image-size 96 `
  --device auto
```

Train the photometric normal-consistency baseline:

```powershell
python scripts/run_feature_baseline.py `
  --data-dir data/raw/cifake `
  --output-dir runs/photometric `
  --feature-set photometric `
  --image-size 128
```

Train the combined conventional baseline:

```powershell
python scripts/run_feature_baseline.py `
  --data-dir data/raw/cifake `
  --output-dir runs/combined_features `
  --feature-set combined `
  --classifier logistic_regression `
  --image-size 128
```

Compare them:

```powershell
python scripts/compare_methods.py `
  --neural runs/resnet18/metrics.json `
  --photometric runs/photometric/metrics.json `
  --out-dir runs/comparison
```

Results are written to:

```text
runs/resnet18/metrics.json
runs/photometric/metrics.json
runs/comparison/report.md
runs/comparison/comparison.csv
```

## Quick Smoke Test

This creates a tiny synthetic dataset and runs both pipelines:

```powershell
python scripts/smoke_test.py
```

It is meant to validate the code path, not measure real accuracy.

## Tests

```powershell
pytest
```

## Using A Different Dataset

Any folder like this should work:

```text
dataset/
  train/
    REAL/
    FAKE/
  test/
    REAL/
    FAKE/
```

or:

```text
dataset/
  REAL/
  FAKE/
```

For a single folder without predefined splits, the scripts make a stratified train/test split.

## Expected Outputs

Each method writes:

- `metrics.json`: accuracy, precision, recall, F1, ROC AUC, confusion matrix.
- `predictions.csv`: per-image or per-sample generated-image score.
- model artifact: `model.pt` for the neural network, `photometric_model.joblib` for the photometric classifier.

The comparison script creates a markdown report and a CSV table so you can track which method wins on the same split.

## Initial CIFAKE Subset Result

An initial balanced CIFAKE subset run is checked into [reports/cifake_subset_initial.md](reports/cifake_subset_initial.md).
It used 2,000 train images and 1,000 test images per method, with a one-epoch ResNet-18 neural baseline.

## Initial 2026 Dataset Result

A full small-dataset run on `ai_vs_real_2026` is checked into [reports/ai_vs_real_2026_benchmark.md](reports/ai_vs_real_2026_benchmark.md).
It used CUDA on the local RTX 3060 Ti and compared photometric, noise, combined conventional features, and pretrained ResNet-18.

## Second 2026 Dataset Result

A second run on `rhythm_ai_vs_real_2026` is checked into [reports/rhythm_ai_vs_real_2026_benchmark.md](reports/rhythm_ai_vs_real_2026_benchmark.md).
It adds a more diverse category split across animals, city, food, nature, and people.
