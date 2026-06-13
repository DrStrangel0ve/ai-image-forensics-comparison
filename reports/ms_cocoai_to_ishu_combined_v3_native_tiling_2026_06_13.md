# MS COCOAI to Ishu combined_v3 Native-Tiling Diagnostic

This bounded diagnostic asks whether the MS-trained `combined_v3` conventional branch benefits from scoring native-resolution target crops instead of only the global resized view. It uses the same saved feature model; each tile is resized to the original feature extraction size before scoring.

## Summary

| score_mode | n_seeds | accuracy_mean | roc_auc_mean | brier_score_mean | expected_calibration_error_mean | predicted_fake_rate_mean | mean_tiles |
| --- | --- | --- | --- | --- | --- | --- | --- |
| global | 3 | 0.5468 | 0.5772 | 0.3078 | 0.2472 | 0.6754 | 6.8772 |
| tile_mean | 3 | 0.5234 | 0.5067 | 0.3167 | 0.2463 | 0.6404 | 6.8772 |
| tile_max | 3 | 0.5468 | 0.5811 | 0.3449 | 0.3096 | 0.8450 | 6.8772 |
| tile_top2_mean | 3 | 0.5468 | 0.5936 | 0.3294 | 0.2880 | 0.8275 | 6.8772 |

## Interpretation

The global resized view reaches 0.5468 mean accuracy / 0.5772 mean AUC. The best default-threshold accuracy mode is `global` at 0.5468; the best ranking mode is `tile_top2_mean` at 0.5936 AUC.
Treat this as a branch-level diagnostic rather than a new SCP-Fusion result: tile aggregation can change sensitivity to local artifacts, but it also changes calibration and fake-call rate before any source-aware thresholding.

## Per-Seed Metrics

| score_mode | seed | accuracy | roc_auc | brier_score | expected_calibration_error | predicted_fake_rate | mean_tiles |
| --- | --- | --- | --- | --- | --- | --- | --- |
| global | 7 | 0.5263 | 0.5727 | 0.3171 | 0.2651 | 0.6842 | 6.8246 |
| global | 17 | 0.5439 | 0.5225 | 0.3374 | 0.2864 | 0.7193 | 6.6316 |
| global | 29 | 0.5702 | 0.6364 | 0.2690 | 0.1901 | 0.6228 | 7.1754 |
| tile_max | 7 | 0.5702 | 0.6056 | 0.3360 | 0.3144 | 0.8158 | 6.8246 |
| tile_max | 17 | 0.5263 | 0.5520 | 0.3574 | 0.3071 | 0.8421 | 6.6316 |
| tile_max | 29 | 0.5439 | 0.5856 | 0.3411 | 0.3074 | 0.8772 | 7.1754 |
| tile_mean | 7 | 0.5175 | 0.5182 | 0.3197 | 0.2496 | 0.6228 | 6.8246 |
| tile_mean | 17 | 0.4912 | 0.4458 | 0.3475 | 0.3008 | 0.6842 | 6.6316 |
| tile_mean | 29 | 0.5614 | 0.5560 | 0.2829 | 0.1886 | 0.6140 | 7.1754 |
| tile_top2_mean | 7 | 0.5439 | 0.6004 | 0.3299 | 0.3063 | 0.7895 | 6.8246 |
| tile_top2_mean | 17 | 0.5351 | 0.5650 | 0.3422 | 0.2841 | 0.8333 | 6.6316 |
| tile_top2_mean | 29 | 0.5614 | 0.6155 | 0.3159 | 0.2735 | 0.8596 | 7.1754 |

## Rebuild

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_tiled_feature_model.py
```
