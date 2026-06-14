# Tiled Frozen-Encoder Evaluation

This report evaluates a saved frozen-encoder classifier with deterministic native-image tile aggregation. The encoder and classifier are unchanged; target images are scored as a global resized view plus native crops.

- model template: `runs\ms_cocoai_to_ishu_foundation\dinov2_vits14_seed{seed}`
- target template: `data\raw\ishu_ai_vs_real_2026_seed{seed}_test_robustness_variants\blur1\ishu_ai_vs_real_2026_seed{seed}_test`
- device: `cuda`
- asset prefix: `tiled_dinov2_blur1_reverse_transfer`

## Summary

| score_mode | n_seeds | accuracy_mean | roc_auc_mean | brier_score_mean | expected_calibration_error_mean | predicted_fake_rate_mean | mean_tiles |
| --- | --- | --- | --- | --- | --- | --- | --- |
| global | 3 | 0.6170 | 0.6639 | 0.3483 | 0.3579 | 0.6813 | 6.8772 |
| tile_mean | 3 | 0.6550 | 0.6806 | 0.2956 | 0.2629 | 0.6842 | 6.8772 |
| tile_max | 3 | 0.5965 | 0.7380 | 0.3651 | 0.3759 | 0.8363 | 6.8772 |
| tile_top2_mean | 3 | 0.6228 | 0.7457 | 0.3484 | 0.3598 | 0.8099 | 6.8772 |

## Interpretation

The global resized view reaches 0.6170 mean accuracy / 0.6639 mean AUC. The best default-threshold accuracy mode is `tile_mean` at 0.6550 (+0.0380 versus global); the best ranking mode is `tile_top2_mean` at 0.7457 AUC (+0.0818 versus global).
Treat this as a foundation-branch diagnostic until the tiled scores are folded into source-heldout SCP-Fusion.

## Per-Seed Metrics

| score_mode | seed | accuracy | roc_auc | brier_score | expected_calibration_error | predicted_fake_rate | mean_tiles |
| --- | --- | --- | --- | --- | --- | --- | --- |
| global | 7 | 0.6491 | 0.6924 | 0.3171 | 0.3303 | 0.6842 | 6.8246 |
| global | 17 | 0.6053 | 0.6610 | 0.3616 | 0.3777 | 0.7105 | 6.6316 |
| global | 29 | 0.5965 | 0.6382 | 0.3663 | 0.3658 | 0.6491 | 7.1754 |
| tile_max | 7 | 0.6228 | 0.7555 | 0.3392 | 0.3551 | 0.8158 | 6.8246 |
| tile_max | 17 | 0.6140 | 0.7289 | 0.3559 | 0.3625 | 0.8246 | 6.6316 |
| tile_max | 29 | 0.5526 | 0.7297 | 0.4003 | 0.4101 | 0.8684 | 7.1754 |
| tile_mean | 7 | 0.6754 | 0.7149 | 0.2659 | 0.2171 | 0.6754 | 6.8246 |
| tile_mean | 17 | 0.6404 | 0.6706 | 0.3139 | 0.2981 | 0.7105 | 6.6316 |
| tile_mean | 29 | 0.6491 | 0.6564 | 0.3070 | 0.2735 | 0.6667 | 7.1754 |
| tile_top2_mean | 7 | 0.6579 | 0.7643 | 0.3171 | 0.3380 | 0.7807 | 6.8246 |
| tile_top2_mean | 17 | 0.6316 | 0.7394 | 0.3449 | 0.3486 | 0.8070 | 6.6316 |
| tile_top2_mean | 29 | 0.5789 | 0.7334 | 0.3832 | 0.3928 | 0.8421 | 7.1754 |

## Rebuild

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_tiled_frozen_encoder_model.py --model-template <run_template> --target-template <target_template>
```
