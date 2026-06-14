# MS COCOAI to Ishu Tuned-Fusion Native-Tiling Diagnostic

This evaluates the selected reverse tuned-fusion operating point after replacing the target-side `convnext_tiny` branch with native-tile aggregate scores. All other branch scores, source training rows, selected fusion configurations, and source threshold policy stay fixed.

## Summary

| score_mode | constraint_policy | n_seeds | target_accuracy_mean | target_roc_auc_mean | target_brier_score_mean | target_expected_calibration_error_mean | target_predicted_positive_rate_mean | source_predicted_positive_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| global | cap_0p4 | 3 | 0.7632 | 0.8361 | 0.2851 | 0.3042 | 0.5175 | 0.4000 |
| tile_mean | cap_0p4 | 3 | 0.7573 | 0.8492 | 0.2715 | 0.2991 | 0.5000 | 0.4000 |
| tile_max | cap_0p4 | 3 | 0.7632 | 0.8414 | 0.2962 | 0.3254 | 0.5409 | 0.4000 |
| tile_top2_mean | cap_0p4 | 3 | 0.7632 | 0.8461 | 0.2910 | 0.3207 | 0.5409 | 0.4000 |

## Clean Comparator

The clean `cap_0p4` tuned-fusion result was 0.7632 accuracy / 0.8361 AUC with a 0.5175 target fake-call rate.
The best native-tiling accuracy mode is `global` at 0.7632; the best ranking mode is `tile_mean` at 0.8492 AUC.
This is a fused-stack diagnostic, not retraining of every visual branch on tiles. It tests whether local crop evidence helps after score fusion when the other branches remain at their normal global-image scores.

## Per-Seed Detail

| score_mode | seed | fusion_c | dropout_config | threshold | target_accuracy | target_roc_auc | target_predicted_positive_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| global | 7 | 0.3000 | mean0p35x8 | 0.9862 | 0.7632 | 0.8381 | 0.5702 |
| global | 17 | 0.0300 | mean0p35x8 | 0.9541 | 0.7105 | 0.7873 | 0.5175 |
| global | 29 | 0.3000 | none | 0.9397 | 0.8158 | 0.8830 | 0.4649 |
| tile_max | 7 | 0.3000 | mean0p35x8 | 0.9862 | 0.7719 | 0.8479 | 0.5789 |
| tile_max | 17 | 0.0300 | mean0p35x8 | 0.9541 | 0.7281 | 0.7980 | 0.5351 |
| tile_max | 29 | 0.3000 | none | 0.9397 | 0.7895 | 0.8784 | 0.5088 |
| tile_mean | 7 | 0.3000 | mean0p35x8 | 0.9862 | 0.7456 | 0.8599 | 0.5351 |
| tile_mean | 17 | 0.0300 | mean0p35x8 | 0.9541 | 0.7105 | 0.7965 | 0.5000 |
| tile_mean | 29 | 0.3000 | none | 0.9397 | 0.8158 | 0.8913 | 0.4649 |
| tile_top2_mean | 7 | 0.3000 | mean0p35x8 | 0.9862 | 0.7719 | 0.8498 | 0.5789 |
| tile_top2_mean | 17 | 0.0300 | mean0p35x8 | 0.9541 | 0.7281 | 0.8045 | 0.5351 |
| tile_top2_mean | 29 | 0.3000 | none | 0.9397 | 0.7895 | 0.8839 | 0.5088 |

## Rebuild

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_reverse_tiled_fusion.py --tile-branch convnext_tiny --tile-detail reports\assets\tiled_convnext_reverse_transfer_detail.csv --asset-prefix ms_cocoai_to_ishu_tuned_fusion_tiled_convnext --report-path reports\ms_cocoai_to_ishu_tuned_fusion_tiled_convnext_2026_06_14.md --alignment-tolerance 0.01
```
