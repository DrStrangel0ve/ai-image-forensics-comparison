# DINOv2 Three-Seed Foundation Baseline

Run date: 2026-06-13

This follow-up upgrades the bounded DINOv2 smoke test into a matched three-seed frozen-encoder baseline. The setup mirrors the existing frozen ConvNeXt-Tiny protocol: full Ishu folder, deterministic 80/20 train/validation split, seeds 7/17/29, frozen embeddings, and a class-balanced logistic-regression head.

## Setup

- encoder: `dinov2_vits14`
- weights: `facebook/dinov2-small`
- embedding dimension: `384`
- classifier: class-balanced logistic regression
- source dataset: Ishu AI-vs-real 2026
- target dataset: source-balanced MS COCOAI validation, `1,000` images
- seeds: `7`, `17`, `29`
- device: `cuda`

## Same-Domain Ishu

| seed | accuracy | AUC | Brier | ECE | precision | recall | F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 0.7719 | 0.8458 | 0.1826 | 0.1893 | 0.7885 | 0.7321 | 0.7593 |
| 17 | 0.8070 | 0.9036 | 0.1435 | 0.1487 | 0.8036 | 0.8036 | 0.8036 |
| 29 | 0.7632 | 0.8350 | 0.1895 | 0.1713 | 0.7377 | 0.8036 | 0.7692 |
| mean | 0.7807 | 0.8615 | 0.1719 | 0.1698 | 0.7766 | 0.7798 | 0.7774 |

DINOv2-small is a real foundation baseline, but it does not beat the existing same-domain leaders. Frozen ConvNeXt-Tiny remains much stronger on Ishu at 0.8947 mean accuracy / 0.9589 mean AUC, and the physics-guided ResNet + `combined_v3` fusion also remains ahead on same-domain accuracy.

## Ishu to MS COCOAI Transfer

| seed | accuracy | AUC | Brier | ECE | precision | recall | F1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 0.6170 | 0.7001 | 0.3348 | 0.3147 | 0.7208 | 0.3820 | 0.4993 |
| 17 | 0.6080 | 0.7023 | 0.3376 | 0.3133 | 0.7143 | 0.3600 | 0.4787 |
| 29 | 0.6150 | 0.7167 | 0.3279 | 0.3082 | 0.7366 | 0.3580 | 0.4818 |
| mean | 0.6133 | 0.7063 | 0.3334 | 0.3121 | 0.7239 | 0.3667 | 0.4866 |

The transfer result is close to frozen ConvNeXt-Tiny's 0.6163 mean accuracy / 0.7139 mean AUC, but with different default-threshold behavior. DINOv2 has lower precision and higher generated-image recall than ConvNeXt, which suggests it may be complementary in score fusion even though it is not a standalone upgrade.

## Target Source Breakdown

Mean over seeds at threshold 0.5:

| source | n per seed | accuracy | mean fake score | detection / FPR |
| --- | ---: | ---: | ---: | ---: |
| real | 500 | 0.8600 | 0.1547 | FPR 0.1400 |
| SD 2.1 | 100 | 0.3100 | 0.2980 | detect 0.3100 |
| SDXL | 100 | 0.4400 | 0.4342 | detect 0.4400 |
| SD3 | 100 | 0.3267 | 0.3420 | detect 0.3267 |
| DALL-E 3 | 100 | 0.4300 | 0.4470 | detect 0.4300 |
| Midjourney 6 | 100 | 0.3267 | 0.3289 | detect 0.3267 |

DINOv2 remains conservative on all generated sources. SDXL and DALL-E 3 are easiest for this branch; SD 2.1, SD3, and Midjourney 6 remain harder.

## Interpretation

This is a useful branch result rather than a leaderboard win:

- DINOv2-small validates the CLIP/DINO foundation-baseline path.
- ConvNeXt remains the stronger same-domain and transfer ranking baseline.
- DINOv2's higher target fake recall may be useful inside SCP-Fusion, where ConvNeXt currently contributes strong precision but conservative fake calls.
- The next experiment should add DINOv2 scores as a fifth SCP-Fusion branch and rerun calibration/triage diagnostics.

## Artifacts

Checked-in compact assets:

- `reports/assets/dinov2_vits14_three_seed_metrics.csv`
- `reports/assets/dinov2_vits14_three_seed_ms_cocoai_source_means.csv`

Local run folders are ignored by Git and can be regenerated under:

- `runs/ishu_ai_vs_real_2026_frozen_encoder/dinov2_vits14_seed{7,17,29}/`
- `runs/ishu_to_ms_cocoai_source_balanced_seed{7,17,29}/dinov2_vits14_frozen/`
- `runs/dinov2_vits14_full/source_analysis_ms_cocoai/`

## Reproduce

Source training/evaluation:

```powershell
python scripts\run_frozen_encoder_baseline.py `
  --data-dir data\raw\ishu_ai_vs_real_2026 `
  --output-dir runs\ishu_ai_vs_real_2026_frozen_encoder\dinov2_vits14_seed7 `
  --encoder dinov2_vits14 `
  --pretrained `
  --classifier logistic_regression `
  --batch-size 32 `
  --num-workers 0 `
  --device cuda `
  --seed 7 `
  --val-fraction 0.2
```

Transfer evaluation:

```powershell
python scripts\evaluate_frozen_encoder_model.py `
  --model-dir runs\ishu_ai_vs_real_2026_frozen_encoder\dinov2_vits14_seed7 `
  --target-dir data\raw\ms_cocoai_2026_validation_source_balanced_100 `
  --output-dir runs\ishu_to_ms_cocoai_source_balanced_seed7\dinov2_vits14_frozen `
  --target-split all `
  --batch-size 32 `
  --num-workers 0 `
  --device cuda
```

Repeat both commands for seeds `17` and `29`.
