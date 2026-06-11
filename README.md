# AI Image Forensics Comparison

This project compares two ways to detect generated images:

1. A standard neural network classifier, defaulting to ResNet-18.
2. A photometric normal-consistency baseline that estimates pseudo normals from each image and checks whether lighting and surface gradients look physically coherent.

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
python - <<'PY'
import torch
print(torch.__version__, torch.cuda.is_available())
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")
PY
```

If PowerShell does not like the heredoc syntax, use:

```powershell
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

## Run The Experiment

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
python scripts/run_photometric_baseline.py `
  --data-dir data/raw/cifake `
  --output-dir runs/photometric `
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
