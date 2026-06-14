# Tiled Frozen-Encoder Evaluation

This report evaluates a saved frozen-encoder classifier with deterministic native-image tile aggregation. The encoder and classifier are unchanged; target images are scored as a global resized view plus native crops.

- model template: `runs\ms_cocoai_to_ishu_foundation\dinov2_vits14_seed{seed}`
- target template: `data/raw/ishu_ai_vs_real_2026_seed{seed}_test`
- device: `cuda`
- asset prefix: `tiled_dinov2_reverse_transfer`

## Summary

| score_mode | n_seeds | accuracy_mean | roc_auc_mean | brier_score_mean | expected_calibration_error_mean | predicted_fake_rate_mean | mean_tiles |
| --- | --- | --- | --- | --- | --- | --- | --- |
| global | 3 | 0.6170 | 0.6669 | 0.3472 | 0.3499 | 0.6871 | 6.8772 |
| tile_mean | 3 | 0.6550 | 0.6818 | 0.2939 | 0.2621 | 0.6842 | 6.8772 |
| tile_max | 3 | 0.6053 | 0.7447 | 0.3645 | 0.3800 | 0.8275 | 6.8772 |
| tile_top2_mean | 3 | 0.6170 | 0.7558 | 0.3497 | 0.3581 | 0.8158 | 6.8772 |

## Interpretation

The global resized view reaches 0.6170 mean accuracy / 0.6669 mean AUC. The best default-threshold accuracy mode is `tile_mean` at 0.6550 (+0.0380 versus global); the best ranking mode is `tile_top2_mean` at 0.7558 AUC (+0.0889 versus global).
Treat this as a foundation-branch diagnostic until the tiled scores are folded into source-heldout SCP-Fusion.

## Per-Seed Metrics

| score_mode | seed | accuracy | roc_auc | brier_score | expected_calibration_error | predicted_fake_rate | mean_tiles |
| --- | --- | --- | --- | --- | --- | --- | --- |
| global | 7 | 0.6754 | 0.7000 | 0.3061 | 0.3178 | 0.6754 | 6.8246 |
| global | 17 | 0.5877 | 0.6629 | 0.3691 | 0.3650 | 0.7281 | 6.6316 |
| global | 29 | 0.5877 | 0.6379 | 0.3663 | 0.3668 | 0.6579 | 7.1754 |
| tile_max | 7 | 0.6491 | 0.7646 | 0.3228 | 0.3395 | 0.7895 | 6.8246 |
| tile_max | 17 | 0.6053 | 0.7264 | 0.3693 | 0.3826 | 0.8333 | 6.6316 |
| tile_max | 29 | 0.5614 | 0.7429 | 0.4013 | 0.4181 | 0.8596 | 7.1754 |
| tile_mean | 7 | 0.7018 | 0.7251 | 0.2561 | 0.2094 | 0.6491 | 6.8246 |
| tile_mean | 17 | 0.6140 | 0.6658 | 0.3190 | 0.3096 | 0.7368 | 6.6316 |
| tile_mean | 29 | 0.6491 | 0.6546 | 0.3067 | 0.2675 | 0.6667 | 7.1754 |
| tile_top2_mean | 7 | 0.6754 | 0.7749 | 0.3043 | 0.3134 | 0.7632 | 6.8246 |
| tile_top2_mean | 17 | 0.6053 | 0.7452 | 0.3585 | 0.3666 | 0.8333 | 6.6316 |
| tile_top2_mean | 29 | 0.5702 | 0.7472 | 0.3864 | 0.3943 | 0.8509 | 7.1754 |

## Rebuild

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_tiled_frozen_encoder_model.py --model-template <run_template> --target-template <target_template>
```
