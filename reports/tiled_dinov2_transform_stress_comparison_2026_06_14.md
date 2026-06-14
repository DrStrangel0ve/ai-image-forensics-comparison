# Tiled DINO Transform Stress Comparison

Run date: 2026-06-14

This compares transformed-target reverse SCP-Fusion robustness before and after replacing the transformed DINOv2-small target branch with native-tile DINOv2 scores. Source rows, selected fusion configurations, and source threshold policy stay fixed.

## Headline

- Largest accuracy lift: `blur1` via `tile_max` (+0.0146 accuracy, +0.0160 AUC).
- Largest AUC lift: `screenshot` via `tile_max` (+0.0163 AUC, +0.0117 accuracy).

Checked transforms show modest accuracy/AUC improvements from tiled-DINO fusion, but calibration is mixed: Brier usually improves for `tile_mean`, while the best accuracy/AUC modes can increase ECE or fake-call rate. Treat this as robustness support for the ranking/decision branch, not a calibration upgrade.

## Comparison

| variant | baseline_accuracy | baseline_auc | best_accuracy_mode | best_accuracy | best_accuracy_auc | accuracy_delta | auc_delta_for_best_accuracy | brier_delta_for_best_accuracy | ece_delta_for_best_accuracy | best_auc_mode | best_auc | auc_delta | standalone_best_accuracy | standalone_best_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blur1 | 0.7105 | 0.7872 | tile_max | 0.7251 | 0.8032 | 0.0146 | 0.0160 | 0.0082 | 0.0194 | tile_max | 0.8032 | 0.0160 | 0.6550 | 0.7457 |
| jpeg30 | 0.7076 | 0.8167 | tile_max | 0.7222 | 0.8292 | 0.0146 | 0.0125 | -0.0032 | 0.0097 | tile_max | 0.8292 | 0.0125 | 0.6579 | 0.7470 |
| resize_half | 0.7164 | 0.7816 | tile_max | 0.7310 | 0.7956 | 0.0146 | 0.0140 | 0.0076 | 0.0126 | tile_max | 0.7956 | 0.0140 | 0.6374 | 0.7412 |
| screenshot | 0.7310 | 0.7965 | tile_max | 0.7427 | 0.8128 | 0.0117 | 0.0163 | 0.0061 | 0.0068 | tile_max | 0.8128 | 0.0163 | 0.6550 | 0.7507 |

## Interpretation

- `tile_max` is currently the strongest transformed-target fusion mode for all checked transforms.
- Standalone tiled DINO remains weak under the checked transforms; the gain appears only when DINO is fused with physical/conventional and other neural/foundation branches.
- The remaining high-resolution tiling gap is no longer these proxy transforms; it is official or paper-compatible external high-resolution benchmark evidence.
