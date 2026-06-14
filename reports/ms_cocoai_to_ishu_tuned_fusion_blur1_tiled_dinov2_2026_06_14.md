# MS COCOAI to Ishu Tuned-Fusion blur1 Tiled-Branch Robustness

This evaluates the already-selected reverse tuned-fusion operating point on `blur1`-transformed Ishu target splits after replacing the transformed `dinov2_vits14` branch with native-tile aggregate scores. Source rows, fusion configuration, and threshold policy stay fixed.

## Summary

| variant | score_mode | constraint_policy | n_seeds | target_accuracy_mean | target_roc_auc_mean | target_brier_score_mean | target_expected_calibration_error_mean | target_predicted_positive_rate_mean | source_predicted_positive_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blur1 | global | cap_0p4 | 3 | 0.7105 | 0.7872 | 0.3280 | 0.3450 | 0.5585 | 0.4000 |
| blur1 | tile_mean | cap_0p4 | 3 | 0.7164 | 0.7914 | 0.3231 | 0.3409 | 0.5585 | 0.4000 |
| blur1 | tile_max | cap_0p4 | 3 | 0.7251 | 0.8032 | 0.3362 | 0.3644 | 0.5848 | 0.4000 |
| blur1 | tile_top2_mean | cap_0p4 | 3 | 0.7222 | 0.8018 | 0.3333 | 0.3622 | 0.5819 | 0.4000 |

## Clean Comparator

The clean `cap_0p4` tuned-fusion result was 0.7632 accuracy / 0.8361 AUC with a 0.5175 target fake-call rate.
The best transformed tiled-branch accuracy mode is `tile_max` at 0.7251; the best ranking mode is `tile_max` at 0.8032 AUC.

## Per-Seed Detail

| variant | score_mode | seed | fusion_c | dropout_config | threshold | target_accuracy | target_roc_auc | target_predicted_positive_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blur1 | global | 7 | 0.3000 | mean0p35x8 | 0.9862 | 0.7018 | 0.8020 | 0.5789 |
| blur1 | global | 17 | 0.0300 | mean0p35x8 | 0.9541 | 0.6754 | 0.7361 | 0.5702 |
| blur1 | global | 29 | 0.3000 | none | 0.9397 | 0.7544 | 0.8233 | 0.5263 |
| blur1 | tile_max | 7 | 0.3000 | mean0p35x8 | 0.9862 | 0.7105 | 0.8150 | 0.6053 |
| blur1 | tile_max | 17 | 0.0300 | mean0p35x8 | 0.9541 | 0.7018 | 0.7669 | 0.6140 |
| blur1 | tile_max | 29 | 0.3000 | none | 0.9397 | 0.7632 | 0.8276 | 0.5351 |
| blur1 | tile_mean | 7 | 0.3000 | mean0p35x8 | 0.9862 | 0.7105 | 0.8070 | 0.5702 |
| blur1 | tile_mean | 17 | 0.0300 | mean0p35x8 | 0.9541 | 0.6754 | 0.7435 | 0.5702 |
| blur1 | tile_mean | 29 | 0.3000 | none | 0.9397 | 0.7632 | 0.8236 | 0.5351 |
| blur1 | tile_top2_mean | 7 | 0.3000 | mean0p35x8 | 0.9862 | 0.7105 | 0.8122 | 0.6053 |
| blur1 | tile_top2_mean | 17 | 0.0300 | mean0p35x8 | 0.9541 | 0.6930 | 0.7660 | 0.6053 |
| blur1 | tile_top2_mean | 29 | 0.3000 | none | 0.9397 | 0.7632 | 0.8273 | 0.5351 |

## Rebuild

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_reverse_tiled_fusion_robustness.py --variant blur1 --tile-branch dinov2_vits14 --tile-detail reports\assets\tiled_dinov2_blur1_reverse_transfer_detail.csv --asset-prefix ms_cocoai_to_ishu_tuned_fusion_blur1_tiled_dinov2 --report-path reports\ms_cocoai_to_ishu_tuned_fusion_blur1_tiled_dinov2_2026_06_14.md
```
