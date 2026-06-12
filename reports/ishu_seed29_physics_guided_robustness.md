# Ishu Seed-29 Physics-Guided Robustness

Run date: 2026-06-12

This check evaluates the seed-29 Ishu test split under common image-sharing transforms. The test images are exported first with `scripts/export_image_split.py`, then transformed with `scripts/make_robustness_variants.py`. This avoids a subtle leakage/error mode where transformed file paths would produce a different hash split.

Clean seed-29 baseline:

| method | accuracy | roc_auc | test images |
| --- | ---: | ---: | ---: |
| `combined_v3` | 0.8333 | 0.8799 | 114 |
| ResNet-18 | 0.8509 | 0.9190 | 114 |
| physics-guided ResNet-18 + `combined_v3` | 0.8772 | 0.9350 | 114 |

Robustness results:

| variant | method | accuracy | accuracy_delta | roc_auc | roc_auc_delta |
| --- | --- | ---: | ---: | ---: | ---: |
| blur1 | `combined_v3` | 0.7895 | -0.0439 | 0.8624 | -0.0175 |
| blur1 | ResNet-18 | 0.8509 | 0.0000 | 0.9212 | 0.0022 |
| blur1 | physics-guided ResNet-18 + `combined_v3` | 0.8860 | 0.0088 | 0.9378 | 0.0028 |
| crop85 | `combined_v3` | 0.8333 | 0.0000 | 0.8858 | 0.0058 |
| crop85 | ResNet-18 | 0.8246 | -0.0263 | 0.8959 | -0.0231 |
| crop85 | physics-guided ResNet-18 + `combined_v3` | 0.8772 | 0.0000 | 0.9280 | -0.0071 |
| jpeg70 | `combined_v3` | 0.8158 | -0.0175 | 0.8772 | -0.0028 |
| jpeg70 | ResNet-18 | 0.8596 | 0.0088 | 0.9200 | 0.0009 |
| jpeg70 | physics-guided ResNet-18 + `combined_v3` | 0.8684 | -0.0088 | 0.9353 | 0.0003 |
| resize_half | `combined_v3` | 0.7895 | -0.0439 | 0.8645 | -0.0154 |
| resize_half | ResNet-18 | 0.8596 | 0.0088 | 0.9224 | 0.0034 |
| resize_half | physics-guided ResNet-18 + `combined_v3` | 0.8772 | 0.0000 | 0.9369 | 0.0018 |

## Interpretation

The fused model stays strongest on every transformed seed-29 split. It is especially encouraging that blur and half-resolution resize do not hurt its AUC; those transforms were more damaging for standalone conventional features on MS COCOAI.

The standalone `combined_v3` model is still the most transform-sensitive branch here, dropping 4.4 accuracy points under both blur and half-resolution resize. ResNet-18 is stable, but the fusion model keeps a small lead over it on every variant.

This is still only one split. The next robustness step is to repeat the same export-and-transform workflow for seeds 7 and 17, then summarize the mean deltas.

## Reproduce

```powershell
python scripts/export_image_split.py `
  --data-dir data/raw/ishu_ai_vs_real_2026 `
  --out-dir data/raw/ishu_ai_vs_real_2026_seed29_test `
  --split test `
  --seed 29 `
  --val-fraction 0.2
```

```powershell
python scripts/make_robustness_variants.py `
  --data-dir data/raw/ishu_ai_vs_real_2026_seed29_test `
  --out-dir data/raw/ishu_ai_vs_real_2026_seed29_test_robustness_variants `
  --variants jpeg70 blur1 resize_half crop85 `
  --format jpg
```

Evaluate each variant folder with `scripts/evaluate_feature_model.py`, `scripts/evaluate_neural_net.py`, and `scripts/evaluate_physics_guided_net.py`, then summarize:

```powershell
python scripts/summarize_robustness_eval.py `
  --out-dir runs/ishu_ai_vs_real_2026_seed29_robustness/summary `
  --baseline combined_v3=runs/ishu_ai_vs_real_2026_repeated_splits_auto/seed29/feature_combined_v3_logistic_regression/metrics.json `
  --baseline resnet18=runs/ishu_ai_vs_real_2026_repeated_splits_auto/seed29/resnet18/metrics.json `
  --baseline physics_guided_resnet18_combined_v3=runs/ishu_ai_vs_real_2026_physics_guided_seed29/physics_guided_resnet18_combined_v3/metrics.json `
  --metrics jpeg70:combined_v3=runs/ishu_ai_vs_real_2026_seed29_robustness/jpeg70/combined_v3/metrics.json `
  --metrics jpeg70:resnet18=runs/ishu_ai_vs_real_2026_seed29_robustness/jpeg70/resnet18/metrics.json `
  --metrics jpeg70:physics_guided_resnet18_combined_v3=runs/ishu_ai_vs_real_2026_seed29_robustness/jpeg70/physics_guided_resnet18_combined_v3/metrics.json `
  --metrics blur1:combined_v3=runs/ishu_ai_vs_real_2026_seed29_robustness/blur1/combined_v3/metrics.json `
  --metrics blur1:resnet18=runs/ishu_ai_vs_real_2026_seed29_robustness/blur1/resnet18/metrics.json `
  --metrics blur1:physics_guided_resnet18_combined_v3=runs/ishu_ai_vs_real_2026_seed29_robustness/blur1/physics_guided_resnet18_combined_v3/metrics.json `
  --metrics resize_half:combined_v3=runs/ishu_ai_vs_real_2026_seed29_robustness/resize_half/combined_v3/metrics.json `
  --metrics resize_half:resnet18=runs/ishu_ai_vs_real_2026_seed29_robustness/resize_half/resnet18/metrics.json `
  --metrics resize_half:physics_guided_resnet18_combined_v3=runs/ishu_ai_vs_real_2026_seed29_robustness/resize_half/physics_guided_resnet18_combined_v3/metrics.json `
  --metrics crop85:combined_v3=runs/ishu_ai_vs_real_2026_seed29_robustness/crop85/combined_v3/metrics.json `
  --metrics crop85:resnet18=runs/ishu_ai_vs_real_2026_seed29_robustness/crop85/resnet18/metrics.json `
  --metrics crop85:physics_guided_resnet18_combined_v3=runs/ishu_ai_vs_real_2026_seed29_robustness/crop85/physics_guided_resnet18_combined_v3/metrics.json
```
