# Tiled DINO Calibration Tradeoff

Run date: 2026-06-14

This report separates the decision/ranking benefit of tiled-DINO branch replacement from its calibration behavior across transformed-target reverse SCP-Fusion probes.

## Headline

- Decision/ranking mode: `tile_max` is the best accuracy mode on all 4 checked transforms; its average deltas are +0.0139 accuracy and +0.0147 AUC.
- Calibration-safe mode: `tile_mean` is the best Brier mode on all checked transforms; `tile_mean` improves Brier on 4/4 transforms and ECE on 3/4 transforms.

The paper-safe claim is therefore two-part: tiled DINO helps source-fixed decisions/ranking when fused, while `tile_mean` is the safer calibration diagnostic and `tile_max` is the stronger operating-point diagnostic.

## Mode Average Deltas

| score_mode | target_accuracy_mean_delta_vs_global | target_roc_auc_mean_delta_vs_global | target_brier_score_mean_delta_vs_global | target_expected_calibration_error_mean_delta_vs_global | target_predicted_positive_rate_mean_delta_vs_global | brier_improvement_count | ece_improvement_count |
| --- | --- | --- | --- | --- | --- | --- | --- |
| global | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 |
| tile_mean | 0.0029 | 0.0040 | -0.0058 | -0.0037 | -0.0029 | 4 | 3 |
| tile_max | 0.0139 | 0.0147 | 0.0047 | 0.0121 | 0.0241 | 1 | 0 |
| tile_top2_mean | 0.0110 | 0.0138 | 0.0024 | 0.0094 | 0.0197 | 1 | 0 |

## Per-Transform Choices

| variant | best_accuracy_mode | best_accuracy_delta | best_accuracy_auc_delta | best_accuracy_brier_delta | best_accuracy_ece_delta | best_brier_mode | best_brier_delta | best_brier_accuracy_delta | best_brier_auc_delta | best_ece_mode | best_ece_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blur1 | tile_max | 0.0146 | 0.0160 | 0.0082 | 0.0194 | tile_mean | -0.0049 | 0.0058 | 0.0042 | tile_mean | -0.0041 |
| jpeg30 | tile_max | 0.0146 | 0.0125 | -0.0032 | 0.0097 | tile_mean | -0.0085 | 0.0000 | 0.0045 | tile_mean | -0.0060 |
| resize_half | tile_max | 0.0146 | 0.0140 | 0.0076 | 0.0126 | tile_mean | -0.0042 | -0.0000 | 0.0028 | global | 0.0000 |
| screenshot | tile_max | 0.0117 | 0.0163 | 0.0061 | 0.0068 | tile_mean | -0.0057 | 0.0058 | 0.0044 | tile_mean | -0.0056 |

## All Mode Deltas

| variant | score_mode | target_accuracy_mean_delta_vs_global | target_roc_auc_mean_delta_vs_global | target_brier_score_mean_delta_vs_global | target_expected_calibration_error_mean_delta_vs_global | target_predicted_positive_rate_mean_delta_vs_global |
| --- | --- | --- | --- | --- | --- | --- |
| blur1 | global | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| blur1 | tile_mean | 0.0058 | 0.0042 | -0.0049 | -0.0041 | 0.0000 |
| blur1 | tile_max | 0.0146 | 0.0160 | 0.0082 | 0.0194 | 0.0263 |
| blur1 | tile_top2_mean | 0.0117 | 0.0147 | 0.0052 | 0.0172 | 0.0234 |
| jpeg30 | global | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| jpeg30 | tile_mean | 0.0000 | 0.0045 | -0.0085 | -0.0060 | -0.0058 |
| jpeg30 | tile_max | 0.0146 | 0.0125 | -0.0032 | 0.0097 | 0.0205 |
| jpeg30 | tile_top2_mean | 0.0088 | 0.0124 | -0.0046 | 0.0080 | 0.0146 |
| resize_half | global | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| resize_half | tile_mean | -0.0000 | 0.0028 | -0.0042 | 0.0011 | -0.0058 |
| resize_half | tile_max | 0.0146 | 0.0140 | 0.0076 | 0.0126 | 0.0263 |
| resize_half | tile_top2_mean | 0.0117 | 0.0138 | 0.0050 | 0.0086 | 0.0234 |
| screenshot | global | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| screenshot | tile_mean | 0.0058 | 0.0044 | -0.0057 | -0.0056 | 0.0000 |
| screenshot | tile_max | 0.0117 | 0.0163 | 0.0061 | 0.0068 | 0.0234 |
| screenshot | tile_top2_mean | 0.0117 | 0.0142 | 0.0039 | 0.0037 | 0.0175 |

## Interpretation

- Use `tile_max` for the robustness headline table, with a calibration caveat.
- Use `tile_mean` when discussing Brier/ECE behavior or calibration-aware operating points.
- Do not claim tiled-DINO improves calibration universally; the ECE result is transform-dependent.
