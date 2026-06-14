# MS COCOAI to Ishu Tuned-Fusion screenshot Tiled-Branch Robustness

This evaluates the already-selected reverse tuned-fusion operating point on `screenshot`-transformed Ishu target splits after replacing the transformed `dinov2_vits14` branch with native-tile aggregate scores. Source rows, fusion configuration, and threshold policy stay fixed.

## Summary

| variant | score_mode | constraint_policy | n_seeds | target_accuracy_mean | target_roc_auc_mean | target_brier_score_mean | target_expected_calibration_error_mean | target_predicted_positive_rate_mean | source_predicted_positive_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| screenshot | global | cap_0p4 | 3 | 0.7310 | 0.7965 | 0.3188 | 0.3392 | 0.5263 | 0.4000 |
| screenshot | tile_mean | cap_0p4 | 3 | 0.7368 | 0.8009 | 0.3131 | 0.3336 | 0.5263 | 0.4000 |
| screenshot | tile_max | cap_0p4 | 3 | 0.7427 | 0.8128 | 0.3249 | 0.3460 | 0.5497 | 0.4000 |
| screenshot | tile_top2_mean | cap_0p4 | 3 | 0.7427 | 0.8107 | 0.3226 | 0.3429 | 0.5439 | 0.4000 |

## Clean Comparator

The clean `cap_0p4` tuned-fusion result was 0.7632 accuracy / 0.8361 AUC with a 0.5175 target fake-call rate.
The best transformed tiled-branch accuracy mode is `tile_max` at 0.7427; the best ranking mode is `tile_max` at 0.8128 AUC.

## Per-Seed Detail

| variant | score_mode | seed | fusion_c | dropout_config | threshold | target_accuracy | target_roc_auc | target_predicted_positive_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| screenshot | global | 7 | 0.3000 | mean0p35x8 | 0.9862 | 0.7281 | 0.8023 | 0.5526 |
| screenshot | global | 17 | 0.0300 | mean0p35x8 | 0.9541 | 0.7018 | 0.7445 | 0.5263 |
| screenshot | global | 29 | 0.3000 | none | 0.9397 | 0.7632 | 0.8427 | 0.5000 |
| screenshot | tile_max | 7 | 0.3000 | mean0p35x8 | 0.9862 | 0.7368 | 0.8140 | 0.5789 |
| screenshot | tile_max | 17 | 0.0300 | mean0p35x8 | 0.9541 | 0.7193 | 0.7783 | 0.5614 |
| screenshot | tile_max | 29 | 0.3000 | none | 0.9397 | 0.7719 | 0.8461 | 0.5088 |
| screenshot | tile_mean | 7 | 0.3000 | mean0p35x8 | 0.9862 | 0.7368 | 0.8073 | 0.5439 |
| screenshot | tile_mean | 17 | 0.0300 | mean0p35x8 | 0.9541 | 0.7018 | 0.7531 | 0.5263 |
| screenshot | tile_mean | 29 | 0.3000 | none | 0.9397 | 0.7719 | 0.8424 | 0.5088 |
| screenshot | tile_top2_mean | 7 | 0.3000 | mean0p35x8 | 0.9862 | 0.7456 | 0.8107 | 0.5702 |
| screenshot | tile_top2_mean | 17 | 0.0300 | mean0p35x8 | 0.9541 | 0.7105 | 0.7752 | 0.5526 |
| screenshot | tile_top2_mean | 29 | 0.3000 | none | 0.9397 | 0.7719 | 0.8461 | 0.5088 |

## Rebuild

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_reverse_tiled_fusion_robustness.py --variant screenshot --tile-branch dinov2_vits14 --tile-detail reports\assets\tiled_dinov2_screenshot_reverse_transfer_detail.csv --asset-prefix ms_cocoai_to_ishu_tuned_fusion_screenshot_tiled_dinov2 --report-path reports\ms_cocoai_to_ishu_tuned_fusion_screenshot_tiled_dinov2_2026_06_14.md
```
