# MS COCOAI to Ishu Tuned-Fusion jpeg30 Tiled-Branch Robustness

This evaluates the already-selected reverse tuned-fusion operating point on `jpeg30`-transformed Ishu target splits after replacing the transformed `dinov2_vits14` branch with native-tile aggregate scores. Source rows, fusion configuration, and threshold policy stay fixed.

## Summary

| variant | score_mode | constraint_policy | n_seeds | target_accuracy_mean | target_roc_auc_mean | target_brier_score_mean | target_expected_calibration_error_mean | target_predicted_positive_rate_mean | source_predicted_positive_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| jpeg30 | global | cap_0p4 | 3 | 0.7076 | 0.8167 | 0.2332 | 0.2215 | 0.3450 | 0.4000 |
| jpeg30 | tile_mean | cap_0p4 | 3 | 0.7076 | 0.8212 | 0.2247 | 0.2155 | 0.3392 | 0.4000 |
| jpeg30 | tile_max | cap_0p4 | 3 | 0.7222 | 0.8292 | 0.2299 | 0.2312 | 0.3655 | 0.4000 |
| jpeg30 | tile_top2_mean | cap_0p4 | 3 | 0.7164 | 0.8291 | 0.2286 | 0.2295 | 0.3596 | 0.4000 |

## Clean Comparator

The clean `cap_0p4` tuned-fusion result was 0.7632 accuracy / 0.8361 AUC with a 0.5175 target fake-call rate.
The best transformed tiled-branch accuracy mode is `tile_max` at 0.7222; the best ranking mode is `tile_max` at 0.8292 AUC.

## Per-Seed Detail

| variant | score_mode | seed | fusion_c | dropout_config | threshold | target_accuracy | target_roc_auc | target_predicted_positive_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| jpeg30 | global | 7 | 0.3000 | mean0p35x8 | 0.9862 | 0.7105 | 0.8374 | 0.4123 |
| jpeg30 | global | 17 | 0.0300 | mean0p35x8 | 0.9541 | 0.6842 | 0.7666 | 0.3509 |
| jpeg30 | global | 29 | 0.3000 | none | 0.9397 | 0.7281 | 0.8461 | 0.2719 |
| jpeg30 | tile_max | 7 | 0.3000 | mean0p35x8 | 0.9862 | 0.7281 | 0.8424 | 0.4298 |
| jpeg30 | tile_max | 17 | 0.0300 | mean0p35x8 | 0.9541 | 0.7018 | 0.7996 | 0.3860 |
| jpeg30 | tile_max | 29 | 0.3000 | none | 0.9397 | 0.7368 | 0.8458 | 0.2807 |
| jpeg30 | tile_mean | 7 | 0.3000 | mean0p35x8 | 0.9862 | 0.7018 | 0.8381 | 0.3860 |
| jpeg30 | tile_mean | 17 | 0.0300 | mean0p35x8 | 0.9541 | 0.6930 | 0.7799 | 0.3596 |
| jpeg30 | tile_mean | 29 | 0.3000 | none | 0.9397 | 0.7281 | 0.8458 | 0.2719 |
| jpeg30 | tile_top2_mean | 7 | 0.3000 | mean0p35x8 | 0.9862 | 0.7193 | 0.8421 | 0.4211 |
| jpeg30 | tile_top2_mean | 17 | 0.0300 | mean0p35x8 | 0.9541 | 0.7018 | 0.7999 | 0.3860 |
| jpeg30 | tile_top2_mean | 29 | 0.3000 | none | 0.9397 | 0.7281 | 0.8454 | 0.2719 |

## Rebuild

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_reverse_tiled_fusion_robustness.py --variant jpeg30 --tile-branch dinov2_vits14 --tile-detail reports\assets\tiled_dinov2_jpeg30_reverse_transfer_detail.csv --asset-prefix ms_cocoai_to_ishu_tuned_fusion_jpeg30_tiled_dinov2 --report-path reports\ms_cocoai_to_ishu_tuned_fusion_jpeg30_tiled_dinov2_2026_06_14.md
```
