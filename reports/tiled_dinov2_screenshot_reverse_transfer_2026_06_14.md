# Tiled Frozen-Encoder Evaluation

This report evaluates a saved frozen-encoder classifier with deterministic native-image tile aggregation. The encoder and classifier are unchanged; target images are scored as a global resized view plus native crops.

- model template: `runs\ms_cocoai_to_ishu_foundation\dinov2_vits14_seed{seed}`
- target template: `data\raw\ishu_ai_vs_real_2026_seed{seed}_test_robustness_variants\screenshot\ishu_ai_vs_real_2026_seed{seed}_test`
- device: `cuda`
- asset prefix: `tiled_dinov2_screenshot_reverse_transfer`

## Summary

| score_mode | n_seeds | accuracy_mean | roc_auc_mean | brier_score_mean | expected_calibration_error_mean | predicted_fake_rate_mean | mean_tiles |
| --- | --- | --- | --- | --- | --- | --- | --- |
| global | 3 | 0.6170 | 0.6638 | 0.3483 | 0.3530 | 0.6754 | 6.8772 |
| tile_mean | 3 | 0.6550 | 0.6859 | 0.2933 | 0.2570 | 0.6784 | 6.8772 |
| tile_max | 3 | 0.6053 | 0.7415 | 0.3611 | 0.3699 | 0.8275 | 6.8772 |
| tile_top2_mean | 3 | 0.6228 | 0.7507 | 0.3461 | 0.3593 | 0.8099 | 6.8772 |

## Interpretation

The global resized view reaches 0.6170 mean accuracy / 0.6638 mean AUC. The best default-threshold accuracy mode is `tile_mean` at 0.6550 (+0.0380 versus global); the best ranking mode is `tile_top2_mean` at 0.7507 AUC (+0.0869 versus global).
Treat this as a foundation-branch diagnostic until the tiled scores are folded into source-heldout SCP-Fusion.

## Per-Seed Metrics

| score_mode | seed | accuracy | roc_auc | brier_score | expected_calibration_error | predicted_fake_rate | mean_tiles |
| --- | --- | --- | --- | --- | --- | --- | --- |
| global | 7 | 0.6491 | 0.6906 | 0.3193 | 0.3221 | 0.6842 | 6.8246 |
| global | 17 | 0.6053 | 0.6639 | 0.3594 | 0.3710 | 0.7105 | 6.6316 |
| global | 29 | 0.5965 | 0.6370 | 0.3661 | 0.3658 | 0.6316 | 7.1754 |
| tile_max | 7 | 0.6404 | 0.7588 | 0.3334 | 0.3460 | 0.7982 | 6.8246 |
| tile_max | 17 | 0.6140 | 0.7268 | 0.3510 | 0.3602 | 0.8246 | 6.6316 |
| tile_max | 29 | 0.5614 | 0.7389 | 0.3988 | 0.4034 | 0.8596 | 7.1754 |
| tile_mean | 7 | 0.6842 | 0.7151 | 0.2683 | 0.2238 | 0.6667 | 6.8246 |
| tile_mean | 17 | 0.6404 | 0.6764 | 0.3089 | 0.2800 | 0.7105 | 6.6316 |
| tile_mean | 29 | 0.6404 | 0.6663 | 0.3028 | 0.2672 | 0.6579 | 7.1754 |
| tile_top2_mean | 7 | 0.6579 | 0.7719 | 0.3176 | 0.3296 | 0.7807 | 6.8246 |
| tile_top2_mean | 17 | 0.6404 | 0.7397 | 0.3414 | 0.3617 | 0.7982 | 6.6316 |
| tile_top2_mean | 29 | 0.5702 | 0.7406 | 0.3793 | 0.3866 | 0.8509 | 7.1754 |

## Rebuild

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_tiled_frozen_encoder_model.py --model-template <run_template> --target-template <target_template>
```
