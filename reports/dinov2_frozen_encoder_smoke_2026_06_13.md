# DINOv2 Frozen Encoder Smoke Probe

Run date: 2026-06-13

This follow-up turns the new CLIP/DINO frozen-encoder support into a first bounded run. It is intentionally small: the classifier sees only 80 Ishu training images and is evaluated on 40 Ishu validation images before transfer to the 1,000-image source-balanced MS COCOAI validation slice.

The purpose is not to claim a final DINOv2 result. It checks whether the new Hugging Face encoder path works end-to-end on the RTX 3060 Ti and whether DINOv2-small is promising enough for a full three-seed run.

## Setup

- encoder: `dinov2_vits14`
- weights: `facebook/dinov2-small`
- embedding dimension: `384`
- classifier: class-balanced logistic regression
- source dataset: Ishu AI-vs-real 2026
- source seed: `7`
- training cap: `80` images
- source validation cap: `40` images
- target dataset: source-balanced MS COCOAI validation, `1,000` images
- device: `cuda`

## Results

| split | accuracy | AUC | Brier | ECE | precision | recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Ishu seed-7 80/40 | 0.7250 | 0.8225 | 0.1812 | 0.1651 | 0.7143 | 0.7500 | 0.7317 |
| Ishu -> MS COCOAI all4 | 0.6080 | 0.7059 | 0.2964 | 0.2566 | 0.7432 | 0.3300 | 0.4571 |

The transfer result is the important signal. Even with only 80 source training examples, DINOv2-small reaches 0.7059 AUC on the source-balanced MS COCOAI target. That is close to the full frozen ConvNeXt-Tiny three-seed transfer mean AUC of 0.7139 reported earlier, though this smoke run has much less source data and only one seed.

At the default 0.5 threshold, the model is conservative: precision is high, but recall is only 0.33 on generated MS COCOAI images.

## Source Breakdown

| source | n | accuracy | mean fake score | detection / FPR |
| --- | ---: | ---: | ---: | ---: |
| real | 500 | 0.8860 | 0.1747 | FPR 0.1140 |
| SD 2.1 | 100 | 0.2700 | 0.3407 | detect 0.2700 |
| SDXL | 100 | 0.3300 | 0.3646 | detect 0.3300 |
| SD3 | 100 | 0.3700 | 0.3912 | detect 0.3700 |
| DALL-E 3 | 100 | 0.3500 | 0.3780 | detect 0.3500 |
| Midjourney 6 | 100 | 0.3300 | 0.3469 | detect 0.3300 |

This mirrors the broader project pattern: pretrained/fusion scores can rank target images reasonably well, but the default threshold under-calls held-out generated images. That makes DINOv2 a strong candidate for SCP-Fusion ranking, but not a standalone thresholded detector yet.

## Artifacts

Checked-in compact assets:

- `reports/assets/dinov2_vits14_smoke_metrics.csv`
- `reports/assets/dinov2_vits14_smoke_ms_cocoai_sources.csv`

Local run folders are ignored by Git and can be regenerated under:

- `runs/frozen_encoder_dinov2_vits14_smoke/ishu_seed7_80_40/`
- `runs/frozen_encoder_dinov2_vits14_smoke/ishu_seed7_80_40_to_ms_cocoai_all4/`
- `runs/frozen_encoder_dinov2_vits14_smoke/source_analysis_ms_cocoai/`

## Reproduce

Install the foundation dependency:

```powershell
.\.venv\Scripts\python -m pip install "transformers>=4.44"
```

Run the bounded source probe:

```powershell
python scripts\run_frozen_encoder_baseline.py `
  --data-dir data\raw\ishu_ai_vs_real_2026 `
  --output-dir runs\frozen_encoder_dinov2_vits14_smoke\ishu_seed7_80_40 `
  --encoder dinov2_vits14 `
  --pretrained `
  --classifier logistic_regression `
  --batch-size 16 `
  --num-workers 0 `
  --device auto `
  --seed 7 `
  --max-train-samples 80 `
  --max-test-samples 40
```

Evaluate transfer:

```powershell
python scripts\evaluate_frozen_encoder_model.py `
  --model-dir runs\frozen_encoder_dinov2_vits14_smoke\ishu_seed7_80_40 `
  --target-dir data\raw\ms_cocoai_2026_validation_source_balanced_100 `
  --output-dir runs\frozen_encoder_dinov2_vits14_smoke\ishu_seed7_80_40_to_ms_cocoai_all4 `
  --target-split all `
  --batch-size 16 `
  --num-workers 0 `
  --device auto
```

Analyze target generators:

```powershell
python scripts\analyze_predictions_by_metadata.py `
  --metadata data\raw\ms_cocoai_2026_validation_source_balanced_100\metadata.csv `
  --predictions dinov2_vits14_smoke=runs\frozen_encoder_dinov2_vits14_smoke\ishu_seed7_80_40_to_ms_cocoai_all4\predictions.csv `
  --split validation `
  --out-dir runs\frozen_encoder_dinov2_vits14_smoke\source_analysis_ms_cocoai
```

## Next Step

Run the full three-seed DINOv2-small baseline with the same sample scale as the ConvNeXt baseline, then add DINOv2 scores into SCP-Fusion and rerun source-heldout calibration/triage.
