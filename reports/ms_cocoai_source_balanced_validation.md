# Defactify / MS COCOAI Source-Balanced Validation

This run reuses the models trained in `runs/ms_cocoai_2026_subset_500` and evaluates them on a newly exported Defactify/MS COCOAI validation slice with balanced generator sources.

Dataset export:

- 500 real validation images.
- 500 generated validation images.
- Generated images are source-balanced: 100 SD2.1, 100 SDXL, 100 SD3, 100 DALL-E 3, and 100 Midjourney 6.
- Source labels come from the Defactify/MS COCOAI `Label_B` metadata column.

Recent dataset context:

- Defactify/MS COCOAI is a recent 96,000-image real-vs-generated dataset with SD2.1, SDXL, SD3, DALL-E 3, and Midjourney v6 sources: https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset
- NTIRE 2026 Robust AI-Generated Image Detection is larger and transformation-heavy, but its validation/test labels are not available for local accuracy scoring yet: https://huggingface.co/datasets/deepfakesMSU/NTIRE-RobustAIGenDetection-train

## Overall Results

| method | accuracy | precision | recall | f1 | roc_auc | n_samples |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| photometric | 0.6820 | 0.6764 | 0.6980 | 0.6870 | 0.7528 | 1000 |
| noise | 0.6770 | 0.6825 | 0.6620 | 0.6721 | 0.7404 | 1000 |
| combined | 0.6960 | 0.6892 | 0.7140 | 0.7014 | 0.7661 | 1000 |
| combined_v2 | 0.7090 | 0.6983 | 0.7360 | 0.7167 | 0.7689 | 1000 |
| resnet18 | 0.8160 | 0.8098 | 0.8260 | 0.8178 | 0.8967 | 1000 |

The stronger conventional baseline is still `combined_v2`, improving over the photometric-only proxy by 2.7 accuracy points. ResNet-18 remains ahead by 10.7 points over `combined_v2`.

## Generator-Level Detection Rate

| method | real false positive rate | SD2.1 | SDXL | SD3 | DALL-E 3 | Midjourney 6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| photometric | 0.3340 | 0.7200 | 0.8900 | 0.6100 | 0.6700 | 0.6000 |
| noise | 0.3080 | 0.7000 | 0.8400 | 0.4000 | 0.7100 | 0.6600 |
| combined | 0.3220 | 0.7000 | 0.9100 | 0.5500 | 0.7800 | 0.6300 |
| combined_v2 | 0.3180 | 0.7000 | 0.9400 | 0.5700 | 0.7900 | 0.6800 |
| resnet18 | 0.1940 | 0.8300 | 0.9000 | 0.6600 | 0.9600 | 0.7800 |

SD3 remains the hardest generated source for both the conventional stack and the neural network. SDXL is easiest for the conventional features, while DALL-E 3 is easiest for ResNet-18.

## Reproduction

```powershell
python scripts/export_hf_image_dataset.py `
  --dataset-key ms_cocoai_2026 `
  --out-dir data/raw/ms_cocoai_2026_validation_source_balanced_100 `
  --splits validation `
  --max-real-per-split 500 `
  --max-per-source-per-split 100 `
  --fake-source-label 1 --fake-source-label 2 --fake-source-label 3 `
  --fake-source-label 4 --fake-source-label 5 `
  --streaming
```

Comparison outputs are under `runs/ms_cocoai_2026_source_balanced_eval`.
