# Tiled Frozen-Encoder Evaluation

This report evaluates a saved frozen-encoder classifier with deterministic native-image tile aggregation. The encoder and classifier are unchanged; target images are scored as a global resized view plus native crops.

- model template: `runs/ms_cocoai_to_ishu_foundation/clip_vit_b_32_seed{seed}`
- target template: `data/raw/ishu_ai_vs_real_2026_seed{seed}_test`
- device: `cuda`
- asset prefix: `tiled_clip_reverse_transfer`

## Summary

| score_mode | n_seeds | accuracy_mean | roc_auc_mean | brier_score_mean | expected_calibration_error_mean | predicted_fake_rate_mean | mean_tiles |
| --- | --- | --- | --- | --- | --- | --- | --- |
| global | 3 | 0.6228 | 0.8244 | 0.3317 | 0.3566 | 0.8626 | 6.8772 |
| tile_mean | 3 | 0.6784 | 0.8030 | 0.2638 | 0.2893 | 0.8012 | 6.8772 |
| tile_max | 3 | 0.5965 | 0.8756 | 0.3602 | 0.3939 | 0.8947 | 6.8772 |
| tile_top2_mean | 3 | 0.6170 | 0.8935 | 0.3328 | 0.3732 | 0.8743 | 6.8772 |

## Interpretation

The global resized view reaches 0.6228 mean accuracy / 0.8244 mean AUC. The best default-threshold accuracy mode is `tile_mean` at 0.6784 (+0.0556 versus global); the best ranking mode is `tile_top2_mean` at 0.8935 AUC (+0.0691 versus global).
Treat this as a foundation-branch diagnostic until the tiled scores are folded into source-heldout SCP-Fusion.

## Per-Seed Metrics

| score_mode | seed | accuracy | roc_auc | brier_score | expected_calibration_error | predicted_fake_rate | mean_tiles |
| --- | --- | --- | --- | --- | --- | --- | --- |
| global | 7 | 0.6140 | 0.8571 | 0.3357 | 0.3674 | 0.8772 | 6.8246 |
| global | 17 | 0.6140 | 0.8059 | 0.3406 | 0.3645 | 0.8772 | 6.6316 |
| global | 29 | 0.6404 | 0.8100 | 0.3189 | 0.3380 | 0.8333 | 7.1754 |
| tile_max | 7 | 0.6140 | 0.9126 | 0.3597 | 0.3918 | 0.8772 | 6.8246 |
| tile_max | 17 | 0.5789 | 0.8578 | 0.3694 | 0.4020 | 0.9123 | 6.6316 |
| tile_max | 29 | 0.5965 | 0.8565 | 0.3513 | 0.3881 | 0.8947 | 7.1754 |
| tile_mean | 7 | 0.6754 | 0.8374 | 0.2752 | 0.3091 | 0.8158 | 6.8246 |
| tile_mean | 17 | 0.6404 | 0.7586 | 0.2882 | 0.3093 | 0.8333 | 6.6316 |
| tile_mean | 29 | 0.7193 | 0.8128 | 0.2278 | 0.2494 | 0.7544 | 7.1754 |
| tile_top2_mean | 7 | 0.6316 | 0.9184 | 0.3375 | 0.3749 | 0.8596 | 6.8246 |
| tile_top2_mean | 17 | 0.5877 | 0.8812 | 0.3501 | 0.3874 | 0.9035 | 6.6316 |
| tile_top2_mean | 29 | 0.6316 | 0.8808 | 0.3110 | 0.3572 | 0.8596 | 7.1754 |

## Rebuild

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_tiled_frozen_encoder_model.py --model-template <run_template> --target-template <target_template>
```
