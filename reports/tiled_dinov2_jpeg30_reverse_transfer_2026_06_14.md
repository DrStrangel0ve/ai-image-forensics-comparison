# Tiled Frozen-Encoder Evaluation

This report evaluates a saved frozen-encoder classifier with deterministic native-image tile aggregation. The encoder and classifier are unchanged; target images are scored as a global resized view plus native crops.

- model template: `runs\ms_cocoai_to_ishu_foundation\dinov2_vits14_seed{seed}`
- target template: `data\raw\ishu_ai_vs_real_2026_seed{seed}_test_robustness_variants\jpeg30\ishu_ai_vs_real_2026_seed{seed}_test`
- device: `cuda`
- asset prefix: `tiled_dinov2_jpeg30_reverse_transfer`

## Summary

| score_mode | n_seeds | accuracy_mean | roc_auc_mean | brier_score_mean | expected_calibration_error_mean | predicted_fake_rate_mean | mean_tiles |
| --- | --- | --- | --- | --- | --- | --- | --- |
| global | 3 | 0.6199 | 0.6585 | 0.3391 | 0.3353 | 0.6433 | 6.8772 |
| tile_mean | 3 | 0.6579 | 0.6801 | 0.2896 | 0.2572 | 0.6520 | 6.8772 |
| tile_max | 3 | 0.6199 | 0.7364 | 0.3523 | 0.3609 | 0.8012 | 6.8772 |
| tile_top2_mean | 3 | 0.6199 | 0.7470 | 0.3377 | 0.3494 | 0.7895 | 6.8772 |

## Interpretation

The global resized view reaches 0.6199 mean accuracy / 0.6585 mean AUC. The best default-threshold accuracy mode is `tile_mean` at 0.6579 (+0.0380 versus global); the best ranking mode is `tile_top2_mean` at 0.7470 AUC (+0.0885 versus global).
Treat this as a foundation-branch diagnostic until the tiled scores are folded into source-heldout SCP-Fusion.

## Per-Seed Metrics

| score_mode | seed | accuracy | roc_auc | brier_score | expected_calibration_error | predicted_fake_rate | mean_tiles |
| --- | --- | --- | --- | --- | --- | --- | --- |
| global | 7 | 0.6754 | 0.6950 | 0.2904 | 0.2890 | 0.6404 | 6.8246 |
| global | 17 | 0.5965 | 0.6529 | 0.3578 | 0.3666 | 0.6842 | 6.6316 |
| global | 29 | 0.5877 | 0.6275 | 0.3692 | 0.3504 | 0.6053 | 7.1754 |
| tile_max | 7 | 0.6491 | 0.7557 | 0.3177 | 0.3213 | 0.7719 | 6.8246 |
| tile_max | 17 | 0.6316 | 0.7288 | 0.3467 | 0.3665 | 0.8070 | 6.6316 |
| tile_max | 29 | 0.5789 | 0.7246 | 0.3925 | 0.3949 | 0.8246 | 7.1754 |
| tile_mean | 7 | 0.7018 | 0.7198 | 0.2599 | 0.2268 | 0.6316 | 6.8246 |
| tile_mean | 17 | 0.6491 | 0.6644 | 0.3082 | 0.2778 | 0.7018 | 6.6316 |
| tile_mean | 29 | 0.6228 | 0.6561 | 0.3008 | 0.2671 | 0.6228 | 7.1754 |
| tile_top2_mean | 7 | 0.6404 | 0.7774 | 0.2965 | 0.3133 | 0.7456 | 6.8246 |
| tile_top2_mean | 17 | 0.6404 | 0.7338 | 0.3361 | 0.3467 | 0.7982 | 6.6316 |
| tile_top2_mean | 29 | 0.5789 | 0.7297 | 0.3804 | 0.3882 | 0.8246 | 7.1754 |

## Rebuild

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_tiled_frozen_encoder_model.py --model-template <run_template> --target-template <target_template>
```
