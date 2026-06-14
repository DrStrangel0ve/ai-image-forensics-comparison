# MS COCOAI to Ishu Tiled Foundation Fusion Comparison

Run date: 2026-06-14

This compares reverse tuned-fusion diagnostics after replacing one target branch at a time with native-tiled target scores. Source training rows, selected fusion configurations, and source threshold policy stay fixed.

## Headline

- Previous fused native-tiling frontier: `combined_v3 conventional` with `tile_max` at `0.7749` accuracy / `0.8472` AUC.
- Best accuracy-preserving tiled foundation replacement: `DINOv2-small` with `tile_top2_mean` at `0.7778` accuracy / `0.8490` AUC.
- Best AUC-only tiled foundation replacement: `ConvNeXt-Tiny` with `tile_mean` at `0.7573` accuracy / `0.8492` AUC.

The gain is real but small: tiled foundation replacement nudges the reverse fused frontier above the previous tiled conventional branch, but it does not close the official SOTA gap and still needs source-heldout and transform stress checks.

## Best Mode Per Replaced Branch

| branch_label | best_accuracy_mode | best_accuracy | accuracy_delta_vs_clean | best_auc_mode | best_auc | auc_delta_vs_clean |
| --- | --- | --- | --- | --- | --- | --- |
| combined_v3 conventional | tile_max | 0.7749 | 0.0117 | tile_max | 0.8472 | 0.0111 |
| CLIP ViT-B/32 | tile_top2_mean | 0.7690 | 0.0058 | tile_top2_mean | 0.8467 | 0.0106 |
| DINOv2-small | tile_top2_mean | 0.7778 | 0.0146 | tile_top2_mean | 0.8490 | 0.0129 |
| ConvNeXt-Tiny | tile_top2_mean | 0.7632 | 0.0000 | tile_mean | 0.8492 | 0.0131 |

## All Score Modes

| branch_label | score_mode | target_accuracy_mean | target_accuracy_mean_delta_vs_clean_global | target_accuracy_mean_delta_vs_previous_tiled_v3 | target_roc_auc_mean | target_roc_auc_mean_delta_vs_clean_global | target_roc_auc_mean_delta_vs_previous_tiled_v3 | target_brier_score_mean | target_expected_calibration_error_mean | target_predicted_positive_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| combined_v3 conventional | global | 0.7632 | 0.0000 | -0.0117 | 0.8361 | 0.0000 | -0.0111 | 0.2851 | 0.3042 | 0.5175 |
| combined_v3 conventional | tile_mean | 0.7749 | 0.0117 | 0.0000 | 0.8326 | -0.0035 | -0.0146 | 0.2854 | 0.3049 | 0.5117 |
| combined_v3 conventional | tile_max | 0.7749 | 0.0117 | 0.0000 | 0.8472 | 0.0111 | 0.0000 | 0.2976 | 0.3188 | 0.5468 |
| combined_v3 conventional | tile_top2_mean | 0.7719 | 0.0088 | -0.0029 | 0.8470 | 0.0109 | -0.0002 | 0.2947 | 0.3154 | 0.5380 |
| CLIP ViT-B/32 | global | 0.7632 | 0.0000 | -0.0117 | 0.8362 | 0.0001 | -0.0110 | 0.2851 | 0.3042 | 0.5175 |
| CLIP ViT-B/32 | tile_mean | 0.7573 | -0.0058 | -0.0175 | 0.8440 | 0.0079 | -0.0032 | 0.2427 | 0.2575 | 0.4883 |
| CLIP ViT-B/32 | tile_max | 0.7661 | 0.0029 | -0.0088 | 0.8405 | 0.0044 | -0.0067 | 0.3024 | 0.3274 | 0.5146 |
| CLIP ViT-B/32 | tile_top2_mean | 0.7690 | 0.0058 | -0.0058 | 0.8467 | 0.0106 | -0.0005 | 0.2865 | 0.3154 | 0.5117 |
| DINOv2-small | global | 0.7632 | 0.0000 | -0.0117 | 0.8361 | 0.0000 | -0.0111 | 0.2851 | 0.3042 | 0.5175 |
| DINOv2-small | tile_mean | 0.7661 | 0.0029 | -0.0088 | 0.8402 | 0.0041 | -0.0070 | 0.2796 | 0.3072 | 0.5146 |
| DINOv2-small | tile_max | 0.7778 | 0.0146 | 0.0029 | 0.8485 | 0.0124 | 0.0013 | 0.2911 | 0.3181 | 0.5439 |
| DINOv2-small | tile_top2_mean | 0.7778 | 0.0146 | 0.0029 | 0.8490 | 0.0129 | 0.0018 | 0.2888 | 0.3150 | 0.5439 |
| ConvNeXt-Tiny | global | 0.7632 | 0.0000 | -0.0117 | 0.8361 | 0.0000 | -0.0111 | 0.2851 | 0.3042 | 0.5175 |
| ConvNeXt-Tiny | tile_mean | 0.7573 | -0.0058 | -0.0175 | 0.8492 | 0.0131 | 0.0021 | 0.2715 | 0.2991 | 0.5000 |
| ConvNeXt-Tiny | tile_max | 0.7632 | 0.0000 | -0.0117 | 0.8414 | 0.0053 | -0.0057 | 0.2962 | 0.3254 | 0.5409 |
| ConvNeXt-Tiny | tile_top2_mean | 0.7632 | 0.0000 | -0.0117 | 0.8461 | 0.0100 | -0.0011 | 0.2910 | 0.3207 | 0.5409 |

## Interpretation

- DINOv2 `tile_top2_mean` gives the best accuracy-first fused operating point.
- ConvNeXt `tile_mean` gives the best AUC-only fused operating point but lowers default accuracy relative to the previous tiled-v3 frontier.
- CLIP tiling helps ranking inside fusion, but its strong standalone tiled AUC does not translate into the best fused branch replacement.
