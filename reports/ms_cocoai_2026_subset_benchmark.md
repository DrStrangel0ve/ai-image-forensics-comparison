# Defactify / MS COCOAI Subset Benchmark

Run date: 2026-06-12

This run benchmarks the existing methods on a newer Hugging Face dataset rather than another Kaggle image-folder dataset. Defactify/MS COCOAI has 96,000 rows, binary real-vs-AI labels, and source labels for Stable Diffusion 2.1, SDXL, Stable Diffusion 3, DALL-E 3, and Midjourney 6.

## Exported Subset

The subset was exported with `scripts/export_hf_image_dataset.py` from `Rajarshi-Roy-research/Defactify_Image_Dataset`.

| split | real | ai_generated | total |
| --- | ---: | ---: | ---: |
| train | 500 | 500 | 1000 |
| validation | 500 | 500 | 1000 |

Synthetic source-label distribution:

| split | SD2.1 | SDXL | SD3 | DALL-E 3 | Midjourney 6 |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 100 | 100 | 100 | 100 | 100 |
| validation | 98 | 85 | 108 | 116 | 93 |

The validation split is class-balanced but not perfectly source-balanced because the streaming export capped binary classes first. That is acceptable for this initial binary benchmark, but a future source-balanced export would be cleaner for per-generator error analysis.

## Results

All methods used the same fixed train/validation split. ResNet-18 used pretrained weights and trained for four epochs on CUDA.

| method | accuracy | precision | recall | f1 | roc_auc | validation images |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| photometric logistic regression | 0.6720 | 0.6700 | 0.6780 | 0.6740 | 0.7413 | 1000 |
| noise logistic regression | 0.6700 | 0.6778 | 0.6480 | 0.6626 | 0.7313 | 1000 |
| combined conventional v1 | 0.6890 | 0.6849 | 0.7000 | 0.6924 | 0.7579 | 1000 |
| combined conventional v2 | 0.7010 | 0.6936 | 0.7200 | 0.7066 | 0.7603 | 1000 |
| ResNet-18 | 0.8160 | 0.8098 | 0.8260 | 0.8178 | 0.8982 | 1000 |

## Interpretation

The neural baseline remains substantially stronger on this subset, with an 11.5 point accuracy lead over the best conventional method and a much larger ROC AUC gap.

The conventional result is still useful: `combined_v2` is the best conventional baseline here, improving accuracy from 0.6890 to 0.7010 over the original combined features. That makes the local noise-entropy additions worth keeping as an experimental robustness baseline.

The photometric proxy alone is close to the noise baseline, but both trail the combined feature sets. That fits the overall pattern of the project: pseudo-normal consistency is a useful signal, not a complete detector for single-image, mixed-generator datasets.

## Source-Level Analysis

The exported metadata includes `Label_B`, so the validation predictions can be grouped by generation source. The table below compares the strongest conventional baseline with ResNet-18.

| source | validation images | combined_v2 detection/FPR | ResNet-18 detection/FPR |
| --- | ---: | ---: | ---: |
| real | 500 | 0.3180 FPR | 0.1940 FPR |
| SD2.1 | 98 | 0.6939 | 0.8265 |
| SDXL | 85 | 0.9294 | 0.8941 |
| SD3 | 108 | 0.5741 | 0.6574 |
| DALL-E 3 | 116 | 0.7586 | 0.9569 |
| Midjourney 6 | 93 | 0.6774 | 0.7957 |

SD3 is the hardest generator family for both detectors. ResNet-18 handles DALL-E 3 especially well on this subset, while `combined_v2` performs best on SDXL but has a high real-image false positive rate.

## Reproduce

```powershell
python scripts/export_hf_image_dataset.py `
  --dataset-key ms_cocoai_2026 `
  --out-dir data/raw/ms_cocoai_2026_subset_500 `
  --splits train validation `
  --max-per-class-per-split 500 `
  --streaming

python scripts/run_benchmark.py `
  --data-dir data/raw/ms_cocoai_2026_subset_500 `
  --out-dir runs/ms_cocoai_2026_subset_500 `
  --methods photometric noise combined combined_v2 neural `
  --feature-classifier logistic_regression `
  --feature-image-size 128 `
  --neural-model resnet18 `
  --pretrained `
  --epochs 4 `
  --batch-size 64 `
  --neural-image-size 128 `
  --num-workers 0 `
  --device cuda

python scripts/analyze_predictions_by_metadata.py `
  --metadata data/raw/ms_cocoai_2026_subset_500/metadata.csv `
  --data-dir data/raw/ms_cocoai_2026_subset_500 `
  --split validation `
  --out-dir runs/ms_cocoai_2026_subset_500/source_analysis `
  --predictions combined_v2=runs/ms_cocoai_2026_subset_500/feature_combined_v2_logistic_regression/predictions.csv `
  --predictions resnet18=runs/ms_cocoai_2026_subset_500/resnet18/predictions.csv
```

## Sources

- [Defactify/MS COCOAI Hugging Face dataset](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset)
- [NTIRE 2026 robust AI-generated image detection training set](https://huggingface.co/datasets/deepfakesMSU/NTIRE-RobustAIGenDetection-train)
