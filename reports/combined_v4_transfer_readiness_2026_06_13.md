# combined_v4 Transfer Readiness

Run date: 2026-06-13

This report records the completed `combined_v4` transfer gate for WIFS/DFF.

## Current Decision

`combined_v4` should remain an ablation candidate for now. In the medium bounded Ishu probe, raw v4 improves AUC over `combined_v3` by 0.0081 and accuracy by 0.0029, but the AUC intervals still overlap. Select-k60 is the calibration-friendly variant, changing ECE by -0.0541 relative to v3 on the same medium probe. The completed full-transfer gate keeps `combined_v3` as the main conventional baseline and moves `combined_v4_selectk60` into the appendix-ablation bucket.

The `combined_v4` transfer rows are present in the core table; the gate outcome is now checked in.

## Delta Versus combined_v3

| scale | candidate | accuracy_delta_vs_v3 | auc_delta_vs_v3 | brier_delta_vs_v3 | ece_delta_vs_v3 | candidate_auc_ci | v3_auc_ci | auc_ci_overlap |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| small_selectk_probe | combined_v4_logreg | -0.0111 | -0.0026 | 0.0005 | 0.0103 | [0.7822, 0.8267] | [0.7511, 0.8300] | True |
| small_selectk_probe | combined_v4_logreg_selectk60 | 0.0111 | 0.0185 | -0.0132 | -0.0003 | [0.8089, 0.8300] | [0.7511, 0.8300] | True |
| medium_240_train_probe | combined_v4_logreg | 0.0029 | 0.0081 | -0.0024 | -0.0141 | [0.8140, 0.8544] | [0.7986, 0.8393] | True |
| medium_240_train_probe | combined_v4_logreg_selectk60 | 0.0000 | 0.0052 | -0.0050 | -0.0541 | [0.8165, 0.8498] | [0.7986, 0.8393] | True |

## Transfer Gate

Promote raw `combined_v4` to the main conventional branch only if the full repeated-seed run shows:

- Ishu same-domain AUC and accuracy stay at or above `combined_v3` without worse calibration.
- Ishu -> MS COCOAI transfer AUC improves over the current `combined_v3` transfer baseline.
- Source-heldout or source-balanced evaluation does not introduce a larger fake-call-rate bias.

Keep select-k60 as a calibration ablation if it keeps lower Brier/ECE even when raw v4 has the better ranking.

## Command Manifest

Commands are written to `reports/assets/combined_v4_transfer_command_manifest.csv` for seeds 7, 17, 29.

The manifest remains checked in for reproduction or extension.

To reproduce the gate:

1. Run all `train` commands.
2. Run all `transfer_eval` commands.
3. Run `python scripts\summarize_combined_v4_transfer.py` and rebuild publication tables.

First command:

```powershell
python scripts\run_feature_baseline.py --data-dir data\raw\ishu_ai_vs_real_2026 --output-dir runs\combined_v4_full_transfer\seed7\combined_v3_logreg --feature-set combined_v3 --classifier logistic_regression --select-k 0 --image-size 128 --seed 7 --val-fraction 0.2 --skip-errors
```

First transfer command:

```powershell
python scripts\evaluate_feature_model.py --model-dir runs\combined_v4_full_transfer\seed7\combined_v3_logreg --target-dir data\raw\ms_cocoai_2026_validation_source_balanced_100 --output-dir runs\combined_v4_full_transfer_to_ms\seed7\combined_v3_logreg --image-size 128 --target-split all --seed 7 --skip-errors
```

## Existing Transfer Rows

| finding_id | setting | method | accuracy | auc | brier | ece |
| --- | --- | --- | --- | --- | --- | --- |
| ishu_to_ms_combined_v4_raw | Ishu -> source-balanced MS COCOAI, combined_v4 transfer gate | combined_v4 | 0.5600 | 0.5801 | 0.3459 | 0.2964 |
| ishu_to_ms_combined_v4_selectk60 | Ishu -> source-balanced MS COCOAI, combined_v4 transfer gate | combined_v4 select-k60 | 0.5553 | 0.5922 | 0.3261 | 0.2663 |
