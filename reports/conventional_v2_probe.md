# Conventional V2 Feature Probe

Run date: 2026-06-12

This probe adds an experimental `combined_v2` feature set. It keeps all original photometric, noise, compression, FFT, ELA, and chroma features, then adds local residual entropy, multiscale residual, Laplacian entropy, and neighboring residual-correlation features. The goal is to approximate more robust noise-artifact checks without making the conventional baseline dependent on a neural model.

The idea is motivated by newer robustness-focused datasets and baselines, including RealHD's noise-entropy direction and NTIRE 2026's focus on transformed in-the-wild generated images.

## Results

| dataset | method | accuracy | precision | recall | f1 | roc_auc | test images |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ai_vs_real_2026` | combined v1 | 0.6821 | 0.7212 | 0.6944 | 0.7075 | 0.7362 | 195 |
| `ai_vs_real_2026` | combined v2 | 0.6718 | 0.7075 | 0.6944 | 0.7009 | 0.7255 | 195 |
| `ai_vs_real_2026` | ResNet-18 | 0.8205 | 0.8288 | 0.8519 | 0.8402 | 0.8992 | 195 |
| `rhythm_ai_vs_real_2026` | combined v1 | 0.8693 | 0.6935 | 0.8600 | 0.7679 | 0.9522 | 199 |
| `rhythm_ai_vs_real_2026` | combined v2 | 0.8894 | 0.7414 | 0.8600 | 0.7963 | 0.9546 | 199 |
| `rhythm_ai_vs_real_2026` | ResNet-18 | 0.9698 | 0.9074 | 0.9800 | 0.9423 | 0.9956 | 199 |

## Interpretation

`combined_v2` is not a universal improvement. It slightly underperforms the original combined baseline on `ai_vs_real_2026`, but improves accuracy, precision, F1, and ROC AUC on `rhythm_ai_vs_real_2026`. The neural ResNet-18 still wins clearly on both same-dataset tests.

For now, keep `combined` as the stable conventional baseline and use `combined_v2` as an experimental baseline when probing robustness. The split-dependent behavior reinforces the cross-dataset finding: conventional signal is useful, but these datasets still have collection-specific artifacts.

## Hugging Face Dataset Export Check

The new Hugging Face exporter was smoke-tested against [Defactify/MS COCOAI](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset) in streaming mode with `--max-per-class-per-split 3`. It produced a valid image-folder sample:

```text
data/raw/ms_cocoai_2026_probe/
  train/
    real/          3 images
    ai_generated/ 3 images
  metadata.csv
```

This validates the path for larger balanced exports from Defactify/MS COCOAI or similar Hugging Face image datasets. A second tiny shuffled streaming export with `--shuffle-buffer 10` also completed, but unauthenticated Hugging Face requests were slow and produced transient retry warnings. Setting `HF_TOKEN` should help with larger exports.

## Reproduce

```powershell
python scripts/run_feature_baseline.py `
  --data-dir data/raw/ai_vs_real_2026 `
  --output-dir runs/ai_vs_real_2026_full/feature_combined_v2_logistic_regression `
  --feature-set combined_v2 `
  --classifier logistic_regression `
  --image-size 128

python scripts/run_feature_baseline.py `
  --data-dir data/raw/rhythm_ai_vs_real_2026 `
  --output-dir runs/rhythm_ai_vs_real_2026_full/feature_combined_v2_logistic_regression `
  --feature-set combined_v2 `
  --classifier logistic_regression `
  --image-size 128

python scripts/export_hf_image_dataset.py `
  --dataset-key ms_cocoai_2026 `
  --out-dir data/raw/ms_cocoai_2026_export `
  --splits train validation `
  --max-per-class-per-split 1000 `
  --shuffle-buffer 1000 `
  --streaming
```
