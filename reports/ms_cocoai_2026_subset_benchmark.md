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
| combined conventional v3 | 0.7290 | 0.7232 | 0.7420 | 0.7325 | 0.7959 | 1000 |
| physics-guided ResNet-18 + `combined_v3` | 0.7810 | 0.7292 | 0.8940 | 0.8032 | 0.8812 | 1000 |
| ResNet-18 | 0.8160 | 0.8098 | 0.8260 | 0.8178 | 0.8982 | 1000 |

## Interpretation

The neural baseline remains strongest on this subset, with a 3.5 point accuracy lead and a 0.0170 ROC AUC lead over the physics-guided fusion model. The fusion model is still a useful middle ground: it improves over standalone `combined_v3` by 5.2 accuracy points and 0.0853 AUC, with much higher generated-image recall.

The conventional result is still useful: `combined_v3` is the best conventional baseline here, improving accuracy from 0.7010 to 0.7290 over `combined_v2`. Its added JPEG recompression, residual periodicity, RGB residual-correlation, and local residual-variance features are worth keeping as the current strongest conventional stack.

The photometric proxy alone is close to the noise baseline, but both trail the combined feature sets. That fits the overall pattern of the project: pseudo-normal consistency is a useful signal, not a complete detector for single-image, mixed-generator datasets.

The fusion result is directional rather than universally better. On Ishu, fusion beat both branches across repeated splits; on MS COCOAI, the vanilla ResNet remains ahead on the same validation images.

## Source-Level Analysis

The exported metadata includes `Label_B`, so the validation predictions can be grouped by generation source. The table below compares the strongest conventional baseline with ResNet-18.

| source | validation images | combined_v2 detection/FPR | combined_v3 detection/FPR | ResNet-18 detection/FPR |
| --- | ---: | ---: | ---: | ---: |
| real | 500 | 0.3180 FPR | 0.2840 FPR | 0.1940 FPR |
| SD2.1 | 98 | 0.6939 | 0.7143 | 0.8265 |
| SDXL | 85 | 0.9294 | 0.8706 | 0.8941 |
| SD3 | 108 | 0.5741 | 0.5741 | 0.6574 |
| DALL-E 3 | 116 | 0.7586 | 0.8534 | 0.9569 |
| Midjourney 6 | 93 | 0.6774 | 0.7097 | 0.7957 |

SD3 is the hardest generator family for both detectors. `combined_v3` lowers the real-image false positive rate and improves DALL-E 3 and Midjourney 6 detection, but gives up some SDXL recall compared with `combined_v2`.

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
  --methods photometric noise combined combined_v2 combined_v3 neural `
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
  --predictions combined_v3=runs/ms_cocoai_2026_subset_500/feature_combined_v3_logistic_regression/predictions.csv `
  --predictions resnet18=runs/ms_cocoai_2026_subset_500/resnet18/predictions.csv

python scripts/train_physics_guided_net.py `
  --data-dir data/raw/ms_cocoai_2026_subset_500 `
  --output-dir runs/ms_cocoai_2026_subset_500/physics_guided_resnet18_combined_v3 `
  --model resnet18 `
  --pretrained `
  --epochs 4 `
  --batch-size 64 `
  --image-size 128 `
  --feature-image-size 128 `
  --num-workers 0 `
  --device cuda `
  --seed 7 `
  --val-fraction 0.2 `
  --physics-feature-set combined_v3 `
  --skip-errors
```

## Sources

- [Defactify/MS COCOAI Hugging Face dataset](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset)
- [NTIRE 2026 robust AI-generated image detection training set](https://huggingface.co/datasets/deepfakesMSU/NTIRE-RobustAIGenDetection-train)
