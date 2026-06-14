# Tiled Frozen-Encoder Evaluation

This report evaluates a saved frozen-encoder classifier with deterministic native-image tile aggregation. The encoder and classifier are unchanged; target images are scored as a global resized view plus native crops.

- model template: `runs\ms_cocoai_to_ishu_foundation\convnext_tiny_seed{seed}`
- target template: `data/raw/ishu_ai_vs_real_2026_seed{seed}_test`
- device: `cuda`
- asset prefix: `tiled_convnext_reverse_transfer`

## Summary

| score_mode | n_seeds | accuracy_mean | roc_auc_mean | brier_score_mean | expected_calibration_error_mean | predicted_fake_rate_mean | mean_tiles |
| --- | --- | --- | --- | --- | --- | --- | --- |
| global | 3 | 0.6579 | 0.8081 | 0.2860 | 0.2950 | 0.7339 | 6.8772 |
| tile_mean | 3 | 0.7310 | 0.8199 | 0.2099 | 0.2135 | 0.6959 | 6.8772 |
| tile_max | 3 | 0.6287 | 0.8602 | 0.3183 | 0.3476 | 0.8392 | 6.8772 |
| tile_top2_mean | 3 | 0.6637 | 0.8762 | 0.2935 | 0.3268 | 0.8041 | 6.8772 |

## Interpretation

The global resized view reaches 0.6579 mean accuracy / 0.8081 mean AUC. The best default-threshold accuracy mode is `tile_mean` at 0.7310 (+0.0731 versus global); the best ranking mode is `tile_top2_mean` at 0.8762 AUC (+0.0680 versus global).
Treat this as a foundation-branch diagnostic until the tiled scores are folded into source-heldout SCP-Fusion.

## Per-Seed Metrics

| score_mode | seed | accuracy | roc_auc | brier_score | expected_calibration_error | predicted_fake_rate | mean_tiles |
| --- | --- | --- | --- | --- | --- | --- | --- |
| global | 7 | 0.6316 | 0.8054 | 0.3025 | 0.3246 | 0.8070 | 6.8246 |
| global | 17 | 0.6316 | 0.7719 | 0.3098 | 0.3138 | 0.7018 | 6.6316 |
| global | 29 | 0.7105 | 0.8471 | 0.2458 | 0.2467 | 0.6930 | 7.1754 |
| tile_max | 7 | 0.6140 | 0.8590 | 0.3332 | 0.3677 | 0.8772 | 6.8246 |
| tile_max | 17 | 0.6228 | 0.8293 | 0.3184 | 0.3396 | 0.7982 | 6.6316 |
| tile_max | 29 | 0.6491 | 0.8922 | 0.3031 | 0.3356 | 0.8421 | 7.1754 |
| tile_mean | 7 | 0.7105 | 0.8076 | 0.2206 | 0.2283 | 0.7456 | 6.8246 |
| tile_mean | 17 | 0.7193 | 0.7651 | 0.2378 | 0.2432 | 0.6491 | 6.6316 |
| tile_mean | 29 | 0.7632 | 0.8870 | 0.1714 | 0.1689 | 0.6930 | 7.1754 |
| tile_top2_mean | 7 | 0.6491 | 0.8773 | 0.3088 | 0.3429 | 0.8421 | 6.8246 |
| tile_top2_mean | 17 | 0.6579 | 0.8341 | 0.2947 | 0.3233 | 0.7632 | 6.6316 |
| tile_top2_mean | 29 | 0.6842 | 0.9172 | 0.2771 | 0.3143 | 0.8070 | 7.1754 |

## Rebuild

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_tiled_frozen_encoder_model.py --model-template <run_template> --target-template <target_template>
```
