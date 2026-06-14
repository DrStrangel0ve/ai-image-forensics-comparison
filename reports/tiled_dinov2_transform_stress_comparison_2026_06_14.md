# Tiled DINO Transform Stress Comparison

Run date: 2026-06-14

This compares transformed-target reverse SCP-Fusion robustness before and after replacing the transformed DINOv2-small target branch with native-tile DINOv2 scores. Source rows, selected fusion configurations, and source threshold policy stay fixed.

## Headline

- Largest accuracy lift: `blur1` via `tile_max` (+0.0146 accuracy, +0.0160 AUC).
- Largest AUC lift: `blur1` via `tile_max` (+0.0160 AUC, +0.0146 accuracy).

Checked transforms show modest accuracy/AUC improvements from tiled-DINO fusion, but calibration is mixed: Brier improves for blur1 and JPEG30 `tile_mean`, while the best accuracy/AUC modes can increase ECE or fake-call rate. Treat this as robustness support for the ranking/decision branch, not a calibration upgrade.

## Comparison

| variant | baseline_accuracy | baseline_auc | best_accuracy_mode | best_accuracy | best_accuracy_auc | accuracy_delta | auc_delta_for_best_accuracy | brier_delta_for_best_accuracy | ece_delta_for_best_accuracy | best_auc_mode | best_auc | auc_delta | standalone_best_accuracy | standalone_best_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blur1 | 0.7105 | 0.7872 | tile_max | 0.7251 | 0.8032 | 0.0146 | 0.0160 | 0.0082 | 0.0194 | tile_max | 0.8032 | 0.0160 | 0.6550 | 0.7457 |
| jpeg30 | 0.7076 | 0.8167 | tile_max | 0.7222 | 0.8292 | 0.0146 | 0.0125 | -0.0032 | 0.0097 | tile_max | 0.8292 | 0.0125 | 0.6579 | 0.7470 |
| resize_half | 0.7164 | 0.7816 | tile_max | 0.7310 | 0.7956 | 0.0146 | 0.0140 | 0.0076 | 0.0126 | tile_max | 0.7956 | 0.0140 | 0.6374 | 0.7412 |

## Interpretation

- `tile_max` is currently the strongest transformed-target fusion mode for all three checked transforms.
- Standalone tiled DINO remains weak under all three checked transforms; the gain appears only when DINO is fused with physical/conventional and other neural/foundation branches.
- Screenshot-style transformed tiled-DINO probes are still needed before promoting this beyond a three-transform diagnostic.
