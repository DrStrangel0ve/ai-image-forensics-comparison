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
- A smaller 2026 Hugging Face candidate, `legenduck/Project1-AI-Generated-Image-Detection-2026`, reports labeled training/validation splits, but the Dataset Viewer first-row API failed with a class-label parsing error during this pass, so it needs a custom import check before benchmarking: https://huggingface.co/datasets/legenduck/Project1-AI-Generated-Image-Detection-2026

The `combined_v3` baseline extends `combined_v2` with JPEG recompression sensitivity, residual 8x8 phase periodicity, RGB residual-correlation, and local residual-variance features. It is still a single-image conventional baseline, not calibrated multi-light photometric stereo.

## Overall Results

| method | accuracy | precision | recall | f1 | roc_auc | n_samples |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| photometric | 0.6820 | 0.6764 | 0.6980 | 0.6870 | 0.7528 | 1000 |
| noise | 0.6770 | 0.6825 | 0.6620 | 0.6721 | 0.7404 | 1000 |
| combined | 0.6960 | 0.6892 | 0.7140 | 0.7014 | 0.7661 | 1000 |
| combined_v2 | 0.7090 | 0.6983 | 0.7360 | 0.7167 | 0.7689 | 1000 |
| combined_v3 | 0.7320 | 0.7248 | 0.7480 | 0.7362 | 0.8027 | 1000 |
| physics_guided_resnet18_combined_v3 | 0.7800 | 0.7288 | 0.8920 | 0.8022 | 0.8790 | 1000 |
| resnet18 | 0.8160 | 0.8098 | 0.8260 | 0.8178 | 0.8967 | 1000 |

The strongest conventional baseline is now `combined_v3`, improving over `combined_v2` by 2.3 accuracy points and over the photometric-only proxy by 5.0 points. The physics-guided fusion model improves over `combined_v3` by 4.8 accuracy points and 0.0763 AUC, but ResNet-18 remains ahead by 3.6 accuracy points and 0.0178 AUC.

## Generator-Level Detection Rate

| method | real false positive rate | SD2.1 | SDXL | SD3 | DALL-E 3 | Midjourney 6 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| photometric | 0.3340 | 0.7200 | 0.8900 | 0.6100 | 0.6700 | 0.6000 |
| noise | 0.3080 | 0.7000 | 0.8400 | 0.4000 | 0.7100 | 0.6600 |
| combined | 0.3220 | 0.7000 | 0.9100 | 0.5500 | 0.7800 | 0.6300 |
| combined_v2 | 0.3180 | 0.7000 | 0.9400 | 0.5700 | 0.7900 | 0.6800 |
| combined_v3 | 0.2840 | 0.7200 | 0.8700 | 0.5700 | 0.8700 | 0.7100 |
| resnet18 | 0.1940 | 0.8300 | 0.9000 | 0.6600 | 0.9600 | 0.7800 |

`combined_v3` reduced real-image false positives from 31.8% to 28.4% versus `combined_v2`, and improved DALL-E 3 and Midjourney 6 detection. It did trade off some SDXL detection, and SD3 remains the hardest generated source for both the conventional stack and the neural network.

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

```powershell
python scripts/run_feature_baseline.py `
  --data-dir data/raw/ms_cocoai_2026_subset_500 `
  --output-dir runs/ms_cocoai_2026_subset_500/feature_combined_v3_logistic_regression `
  --feature-set combined_v3 `
  --classifier logistic_regression `
  --image-size 128

python scripts/evaluate_feature_model.py `
  --model-dir runs/ms_cocoai_2026_subset_500/feature_combined_v3_logistic_regression `
  --target-dir data/raw/ms_cocoai_2026_validation_source_balanced_100 `
  --output-dir runs/ms_cocoai_2026_source_balanced_eval/combined_v3 `
  --image-size 128 `
  --target-split all

python scripts/evaluate_physics_guided_net.py `
  --model-dir runs/ms_cocoai_2026_subset_500/physics_guided_resnet18_combined_v3 `
  --target-dir data/raw/ms_cocoai_2026_validation_source_balanced_100 `
  --output-dir runs/ms_cocoai_2026_source_balanced_eval/physics_guided_resnet18_combined_v3 `
  --image-size 128 `
  --feature-image-size 128 `
  --batch-size 64 `
  --num-workers 0 `
  --device cuda `
  --target-split all `
  --skip-errors
```

Comparison outputs are under `runs/ms_cocoai_2026_source_balanced_eval`.
