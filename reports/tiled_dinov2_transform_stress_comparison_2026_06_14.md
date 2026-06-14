# Tiled DINO Transform Stress Comparison

Run date: 2026-06-14

This compares transformed-target reverse SCP-Fusion robustness before and after replacing the transformed DINOv2-small target branch with native-tile DINOv2 scores. Source rows, selected fusion configurations, and source threshold policy stay fixed.

## Headline

- Largest accuracy lift: `blur1` via `tile_max` (+0.0146 accuracy, +0.0160 AUC).
- Largest AUC lift: `blur1` via `tile_max` (+0.0160 AUC, +0.0146 accuracy).

Both checked transforms show modest accuracy/AUC improvements from tiled-DINO fusion, but calibration is mixed: Brier improves for blur1 `tile_mean`, while the best accuracy/AUC modes increase ECE and fake-call rate. Treat this as robustness support for the ranking/decision branch, not a calibration upgrade.

## Comparison

| variant | baseline_accuracy | baseline_auc | best_accuracy_mode | best_accuracy | best_accuracy_auc | accuracy_delta | auc_delta_for_best_accuracy | brier_delta_for_best_accuracy | ece_delta_for_best_accuracy | best_auc_mode | best_auc | auc_delta | standalone_best_accuracy | standalone_best_auc |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| blur1 | 0.7105 | 0.7872 | tile_max | 0.7251 | 0.8032 | 0.0146 | 0.0160 | 0.0082 | 0.0194 | tile_max | 0.8032 | 0.0160 | 0.6550 | 0.7457 |
| resize_half | 0.7164 | 0.7816 | tile_max | 0.7310 | 0.7956 | 0.0146 | 0.0140 | 0.0076 | 0.0126 | tile_max | 0.7956 | 0.0140 | 0.6374 | 0.7412 |

## Interpretation

- `tile_max` is currently the strongest transformed-target fusion mode for both blur1 and resize-half.
- Standalone tiled DINO remains weak under both transforms; the gain appears only when DINO is fused with physical/conventional and other neural/foundation branches.
- JPEG and screenshot-style transformed tiled-DINO probes are still needed before promoting this beyond a two-transform diagnostic.
