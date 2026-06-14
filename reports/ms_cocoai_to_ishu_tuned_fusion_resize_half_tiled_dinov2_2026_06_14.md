# MS COCOAI to Ishu Tuned-Fusion resize_half Tiled-Branch Robustness

This evaluates the already-selected reverse tuned-fusion operating point on `resize_half`-transformed Ishu target splits after replacing the transformed `dinov2_vits14` branch with native-tile aggregate scores. Source rows, fusion configuration, and threshold policy stay fixed.

## Summary

| variant | score_mode | constraint_policy | n_seeds | target_accuracy_mean | target_roc_auc_mean | target_brier_score_mean | target_expected_calibration_error_mean | target_predicted_positive_rate_mean | source_predicted_positive_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| resize_half | global | cap_0p4 | 3 | 0.7164 | 0.7816 | 0.3249 | 0.3410 | 0.5526 | 0.4000 |
| resize_half | tile_mean | cap_0p4 | 3 | 0.7164 | 0.7844 | 0.3206 | 0.3421 | 0.5468 | 0.4000 |
| resize_half | tile_max | cap_0p4 | 3 | 0.7310 | 0.7956 | 0.3325 | 0.3536 | 0.5789 | 0.4000 |
| resize_half | tile_top2_mean | cap_0p4 | 3 | 0.7281 | 0.7954 | 0.3299 | 0.3496 | 0.5760 | 0.4000 |

## Clean Comparator

The clean `cap_0p4` tuned-fusion result was 0.7632 accuracy / 0.8361 AUC with a 0.5175 target fake-call rate.
The best transformed tiled-branch accuracy mode is `tile_max` at 0.7310; the best ranking mode is `tile_max` at 0.7956 AUC.

## Per-Seed Detail

| variant | score_mode | seed | fusion_c | dropout_config | threshold | target_accuracy | target_roc_auc | target_predicted_positive_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| resize_half | global | 7 | 0.3000 | mean0p35x8 | 0.9862 | 0.7018 | 0.7953 | 0.5789 |
| resize_half | global | 17 | 0.0300 | mean0p35x8 | 0.9541 | 0.6930 | 0.7254 | 0.5526 |
| resize_half | global | 29 | 0.3000 | none | 0.9397 | 0.7544 | 0.8242 | 0.5263 |
| resize_half | tile_max | 7 | 0.3000 | mean0p35x8 | 0.9862 | 0.7105 | 0.8039 | 0.6053 |
| resize_half | tile_max | 17 | 0.0300 | mean0p35x8 | 0.9541 | 0.7193 | 0.7534 | 0.5965 |
| resize_half | tile_max | 29 | 0.3000 | none | 0.9397 | 0.7632 | 0.8294 | 0.5351 |
| resize_half | tile_mean | 7 | 0.3000 | mean0p35x8 | 0.9862 | 0.7105 | 0.8014 | 0.5702 |
| resize_half | tile_mean | 17 | 0.0300 | mean0p35x8 | 0.9541 | 0.6930 | 0.7251 | 0.5526 |
| resize_half | tile_mean | 29 | 0.3000 | none | 0.9397 | 0.7456 | 0.8267 | 0.5175 |
| resize_half | tile_top2_mean | 7 | 0.3000 | mean0p35x8 | 0.9862 | 0.7105 | 0.8036 | 0.6053 |
| resize_half | tile_top2_mean | 17 | 0.0300 | mean0p35x8 | 0.9541 | 0.7105 | 0.7534 | 0.5877 |
| resize_half | tile_top2_mean | 29 | 0.3000 | none | 0.9397 | 0.7632 | 0.8291 | 0.5351 |

## Rebuild

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_reverse_tiled_fusion_robustness.py --variant resize_half --tile-branch dinov2_vits14 --tile-detail reports\assets\tiled_dinov2_resize_half_reverse_transfer_detail.csv --asset-prefix ms_cocoai_to_ishu_tuned_fusion_resize_half_tiled_dinov2 --report-path reports\ms_cocoai_to_ishu_tuned_fusion_resize_half_tiled_dinov2_2026_06_14.md
```
