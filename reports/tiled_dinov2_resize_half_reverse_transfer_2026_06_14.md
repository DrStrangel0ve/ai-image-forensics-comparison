# Tiled Frozen-Encoder Evaluation

This report evaluates a saved frozen-encoder classifier with deterministic native-image tile aggregation. The encoder and classifier are unchanged; target images are scored as a global resized view plus native crops.

- model template: `runs\ms_cocoai_to_ishu_foundation\dinov2_vits14_seed{seed}`
- target template: `data\raw\ishu_ai_vs_real_2026_seed{seed}_test_robustness_variants\resize_half\ishu_ai_vs_real_2026_seed{seed}_test`
- device: `cuda`
- asset prefix: `tiled_dinov2_resize_half_reverse_transfer`

## Summary

| score_mode | n_seeds | accuracy_mean | roc_auc_mean | brier_score_mean | expected_calibration_error_mean | predicted_fake_rate_mean | mean_tiles |
| --- | --- | --- | --- | --- | --- | --- | --- |
| global | 3 | 0.6082 | 0.6569 | 0.3538 | 0.3566 | 0.6901 | 6.8772 |
| tile_mean | 3 | 0.6374 | 0.6697 | 0.3024 | 0.2694 | 0.7018 | 6.8772 |
| tile_max | 3 | 0.5906 | 0.7316 | 0.3701 | 0.3844 | 0.8421 | 6.8772 |
| tile_top2_mean | 3 | 0.6111 | 0.7412 | 0.3541 | 0.3649 | 0.8216 | 6.8772 |

## Interpretation

The global resized view reaches 0.6082 mean accuracy / 0.6569 mean AUC. The best default-threshold accuracy mode is `tile_mean` at 0.6374 (+0.0292 versus global); the best ranking mode is `tile_top2_mean` at 0.7412 AUC (+0.0843 versus global).
Treat this as a foundation-branch diagnostic until the tiled scores are folded into source-heldout SCP-Fusion.

## Per-Seed Metrics

| score_mode | seed | accuracy | roc_auc | brier_score | expected_calibration_error | predicted_fake_rate | mean_tiles |
| --- | --- | --- | --- | --- | --- | --- | --- |
| global | 7 | 0.6404 | 0.6775 | 0.3268 | 0.3405 | 0.6930 | 6.8246 |
| global | 17 | 0.5789 | 0.6567 | 0.3687 | 0.3674 | 0.7368 | 6.6316 |
| global | 29 | 0.6053 | 0.6365 | 0.3661 | 0.3620 | 0.6404 | 7.1754 |
| tile_max | 7 | 0.6053 | 0.7486 | 0.3482 | 0.3629 | 0.8333 | 6.8246 |
| tile_max | 17 | 0.5877 | 0.7218 | 0.3645 | 0.3829 | 0.8509 | 6.6316 |
| tile_max | 29 | 0.5789 | 0.7244 | 0.3976 | 0.4073 | 0.8421 | 7.1754 |
| tile_mean | 7 | 0.6754 | 0.7024 | 0.2761 | 0.2416 | 0.6930 | 6.8246 |
| tile_mean | 17 | 0.6053 | 0.6547 | 0.3235 | 0.2954 | 0.7456 | 6.6316 |
| tile_mean | 29 | 0.6316 | 0.6521 | 0.3077 | 0.2712 | 0.6667 | 7.1754 |
| tile_top2_mean | 7 | 0.6491 | 0.7591 | 0.3256 | 0.3432 | 0.7895 | 6.8246 |
| tile_top2_mean | 17 | 0.6053 | 0.7323 | 0.3536 | 0.3635 | 0.8333 | 6.6316 |
| tile_top2_mean | 29 | 0.5789 | 0.7323 | 0.3831 | 0.3880 | 0.8421 | 7.1754 |

## Rebuild

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_tiled_frozen_encoder_model.py --model-template <run_template> --target-template <target_template>
```
