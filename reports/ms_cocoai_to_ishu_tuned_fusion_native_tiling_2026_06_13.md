# MS COCOAI to Ishu Tuned-Fusion Native-Tiling Diagnostic

This evaluates the selected reverse tuned-fusion operating point after replacing the target-side `combined_v3` branch with native-tile aggregate scores. All other branch scores, source training rows, selected fusion configurations, and source threshold policy stay fixed.

## Summary

| score_mode | constraint_policy | n_seeds | target_accuracy_mean | target_roc_auc_mean | target_brier_score_mean | target_expected_calibration_error_mean | target_predicted_positive_rate_mean | source_predicted_positive_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| global | cap_0p4 | 3 | 0.7632 | 0.8361 | 0.2851 | 0.3042 | 0.5175 | 0.4000 |
| tile_mean | cap_0p4 | 3 | 0.7749 | 0.8326 | 0.2854 | 0.3049 | 0.5117 | 0.4000 |
| tile_max | cap_0p4 | 3 | 0.7749 | 0.8472 | 0.2976 | 0.3188 | 0.5468 | 0.4000 |
| tile_top2_mean | cap_0p4 | 3 | 0.7719 | 0.8470 | 0.2947 | 0.3154 | 0.5380 | 0.4000 |

## Clean Comparator

The clean `cap_0p4` tuned-fusion result was 0.7632 accuracy / 0.8361 AUC with a 0.5175 target fake-call rate.
The best native-tiling accuracy mode is `tile_mean` at 0.7749; the best ranking mode is `tile_max` at 0.8472 AUC.
This is a fused-stack diagnostic, not retraining of every visual branch on tiles. It tests whether local crop evidence helps after score fusion when the other branches remain at their normal global-image scores.

## Per-Seed Detail

| score_mode | seed | fusion_c | dropout_config | threshold | target_accuracy | target_roc_auc | target_predicted_positive_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| global | 7 | 0.3000 | mean0p35x8 | 0.9862 | 0.7632 | 0.8381 | 0.5702 |
| global | 17 | 0.0300 | mean0p35x8 | 0.9541 | 0.7105 | 0.7873 | 0.5175 |
| global | 29 | 0.3000 | none | 0.9397 | 0.8158 | 0.8830 | 0.4649 |
| tile_max | 7 | 0.3000 | mean0p35x8 | 0.9862 | 0.7719 | 0.8461 | 0.5789 |
| tile_max | 17 | 0.0300 | mean0p35x8 | 0.9541 | 0.7193 | 0.7999 | 0.5439 |
| tile_max | 29 | 0.3000 | none | 0.9397 | 0.8333 | 0.8956 | 0.5175 |
| tile_mean | 7 | 0.3000 | mean0p35x8 | 0.9862 | 0.7632 | 0.8328 | 0.5526 |
| tile_mean | 17 | 0.0300 | mean0p35x8 | 0.9541 | 0.7193 | 0.7842 | 0.5263 |
| tile_mean | 29 | 0.3000 | none | 0.9397 | 0.8421 | 0.8808 | 0.4561 |
| tile_top2_mean | 7 | 0.3000 | mean0p35x8 | 0.9862 | 0.7632 | 0.8458 | 0.5702 |
| tile_top2_mean | 17 | 0.0300 | mean0p35x8 | 0.9541 | 0.7193 | 0.7980 | 0.5439 |
| tile_top2_mean | 29 | 0.3000 | none | 0.9397 | 0.8333 | 0.8972 | 0.5000 |

## Rebuild

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_reverse_tiled_fusion.py --tile-branch combined_v3 --tile-detail reports/assets/ms_cocoai_to_ishu_combined_v3_native_tiling_detail.csv --asset-prefix ms_cocoai_to_ishu_tuned_fusion_native_tiling --report-path reports/ms_cocoai_to_ishu_tuned_fusion_native_tiling_2026_06_13.md
```
