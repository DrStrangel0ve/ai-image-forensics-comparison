# Tiled Foundation Reverse-Transfer Comparison

Run date: 2026-06-14

This report compares native-image tile aggregation for the three saved MS-COCOAI-to-Ishu frozen-encoder classifiers. The classifiers are unchanged; only target-side scoring changes from one global resized image to deterministic native tiles.

## Headline

- Best default-threshold accuracy: `ConvNeXt-Tiny` with `tile_mean` at `0.7310` accuracy.
- Best ranking AUC: `CLIP ViT-B/32` with `tile_top2_mean` at `0.8935` AUC.
- Best Brier score: `ConvNeXt-Tiny` with `tile_mean` at `0.2099`.
- Best ECE: `ConvNeXt-Tiny` with `tile_mean` at `0.2135`.

ConvNeXt-Tiny is the strongest tiled forced-decision encoder in this reverse direction, while CLIP remains the strongest ranking encoder. DINOv2 improves under tiling, but its reverse-transfer global baseline is too weak to catch the other two encoders.

## Best Mode Per Encoder

| encoder | best_accuracy_mode | best_accuracy | accuracy_delta_vs_global | best_auc_mode | best_auc | auc_delta_vs_global | best_brier_for_accuracy_mode | source_report |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CLIP ViT-B/32 | tile_mean | 0.6784 | 0.0556 | tile_top2_mean | 0.8935 | 0.0691 | 0.2638 | reports/tiled_clip_reverse_transfer_2026_06_14.md |
| DINOv2-small | tile_mean | 0.6550 | 0.0380 | tile_top2_mean | 0.7558 | 0.0889 | 0.2939 | reports/tiled_dinov2_reverse_transfer_2026_06_14.md |
| ConvNeXt-Tiny | tile_mean | 0.7310 | 0.0731 | tile_top2_mean | 0.8762 | 0.0680 | 0.2099 | reports/tiled_convnext_reverse_transfer_2026_06_14.md |

## All Score Modes

| encoder | score_mode | accuracy_mean | accuracy_mean_delta_vs_global | roc_auc_mean | roc_auc_mean_delta_vs_global | brier_score_mean | expected_calibration_error_mean | predicted_fake_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CLIP ViT-B/32 | global | 0.6228 | 0.0000 | 0.8244 | 0.0000 | 0.3317 | 0.3566 | 0.8626 |
| CLIP ViT-B/32 | tile_mean | 0.6784 | 0.0556 | 0.8030 | -0.0214 | 0.2638 | 0.2893 | 0.8012 |
| CLIP ViT-B/32 | tile_max | 0.5965 | -0.0263 | 0.8756 | 0.0513 | 0.3602 | 0.3939 | 0.8947 |
| CLIP ViT-B/32 | tile_top2_mean | 0.6170 | -0.0058 | 0.8935 | 0.0691 | 0.3328 | 0.3732 | 0.8743 |
| DINOv2-small | global | 0.6170 | 0.0000 | 0.6669 | 0.0000 | 0.3472 | 0.3499 | 0.6871 |
| DINOv2-small | tile_mean | 0.6550 | 0.0380 | 0.6818 | 0.0149 | 0.2939 | 0.2621 | 0.6842 |
| DINOv2-small | tile_max | 0.6053 | -0.0117 | 0.7447 | 0.0777 | 0.3645 | 0.3800 | 0.8275 |
| DINOv2-small | tile_top2_mean | 0.6170 | 0.0000 | 0.7558 | 0.0889 | 0.3497 | 0.3581 | 0.8158 |
| ConvNeXt-Tiny | global | 0.6579 | 0.0000 | 0.8081 | 0.0000 | 0.2860 | 0.2950 | 0.7339 |
| ConvNeXt-Tiny | tile_mean | 0.7310 | 0.0731 | 0.8199 | 0.0118 | 0.2099 | 0.2135 | 0.6959 |
| ConvNeXt-Tiny | tile_max | 0.6287 | -0.0292 | 0.8602 | 0.0520 | 0.3183 | 0.3476 | 0.8392 |
| ConvNeXt-Tiny | tile_top2_mean | 0.6637 | 0.0058 | 0.8762 | 0.0680 | 0.2935 | 0.3268 | 0.8041 |

## Interpretation

- `tile_mean` is the safer operating-point aggregator: it improves accuracy/calibration for all three encoders.
- `tile_top2_mean` is the stronger ranking aggregator for all three encoders, but it tends to raise the predicted fake rate and should be threshold-calibrated before deployment claims.
- The next SCP-Fusion step is to feed the best tiled foundation score modes into the reverse fusion stack and test whether they beat the current native-tiled conventional branch result of 0.7749 accuracy / 0.8472 AUC.
