# combined_v4 Full-Transfer Summary

Run date: 2026-06-13

This report summarizes the checked local `metrics.json` files for the `combined_v4` transfer gate: Ishu same-domain holdout plus Ishu -> source-balanced MS COCOAI transfer.

Seeds included: 7. This is a preliminary seed slice, not a promotion decision.

## Seed Results

| phase_label | seed | run | accuracy | roc_auc | brier_score | expected_calibration_error | fake_call_rate | real_false_positive_rate | fake_miss_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ishu holdout split | 7 | combined_v3_logreg | 0.8158 | 0.8938 | 0.1324 | 0.0807 | 0.5000 | 0.1897 | 0.1786 |
| Ishu holdout split | 7 | combined_v4_logreg | 0.8070 | 0.8990 | 0.1340 | 0.0752 | 0.4737 | 0.1724 | 0.2143 |
| Ishu holdout split | 7 | combined_v4_logreg_selectk60 | 0.7982 | 0.8867 | 0.1373 | 0.0740 | 0.4825 | 0.1897 | 0.2143 |
| Ishu -> source-balanced MS COCOAI | 7 | combined_v3_logreg | 0.5430 | 0.5750 | 0.3435 | 0.2969 | 0.1790 | 0.1360 | 0.7780 |
| Ishu -> source-balanced MS COCOAI | 7 | combined_v4_logreg | 0.5520 | 0.5685 | 0.3483 | 0.2910 | 0.1960 | 0.1440 | 0.7520 |
| Ishu -> source-balanced MS COCOAI | 7 | combined_v4_logreg_selectk60 | 0.5590 | 0.5976 | 0.3154 | 0.2470 | 0.2050 | 0.1460 | 0.7360 |

## Mean Results

| phase_label | run | n_seeds | accuracy_mean | roc_auc_mean | brier_score_mean | expected_calibration_error_mean | fake_call_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ishu holdout split | combined_v3_logreg | 1 | 0.8158 | 0.8938 | 0.1324 | 0.0807 | 0.5000 |
| Ishu holdout split | combined_v4_logreg | 1 | 0.8070 | 0.8990 | 0.1340 | 0.0752 | 0.4737 |
| Ishu holdout split | combined_v4_logreg_selectk60 | 1 | 0.7982 | 0.8867 | 0.1373 | 0.0740 | 0.4825 |
| Ishu -> source-balanced MS COCOAI | combined_v3_logreg | 1 | 0.5430 | 0.5750 | 0.3435 | 0.2969 | 0.1790 |
| Ishu -> source-balanced MS COCOAI | combined_v4_logreg | 1 | 0.5520 | 0.5685 | 0.3483 | 0.2910 | 0.1960 |
| Ishu -> source-balanced MS COCOAI | combined_v4_logreg_selectk60 | 1 | 0.5590 | 0.5976 | 0.3154 | 0.2470 | 0.2050 |

## Paired Delta Versus combined_v3

| phase_label | candidate | n_paired_seeds | accuracy_delta_mean | roc_auc_delta_mean | brier_score_delta_mean | expected_calibration_error_delta_mean | fake_call_rate_delta_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Ishu holdout split | combined_v4_logreg | 1 | -0.0088 | 0.0052 | 0.0016 | -0.0055 | -0.0263 |
| Ishu holdout split | combined_v4_logreg_selectk60 | 1 | -0.0175 | -0.0071 | 0.0049 | -0.0067 | -0.0175 |
| Ishu -> source-balanced MS COCOAI | combined_v4_logreg | 1 | 0.0090 | -0.0065 | 0.0048 | -0.0059 | 0.0170 |
| Ishu -> source-balanced MS COCOAI | combined_v4_logreg_selectk60 | 1 | 0.0160 | 0.0226 | -0.0281 | -0.0499 | 0.0260 |

## Interpretation

For the current seed slice, `combined_v4_selectk60` is the most interesting transfer candidate because it improves MS COCOAI transfer accuracy, AUC, Brier score, and ECE versus `combined_v3`. The same model is slightly weaker on the Ishu holdout split, so it should stay an ablation until seeds 17 and 29 confirm whether this is a real cross-domain gain.

Raw `combined_v4` remains a useful diagnostic branch: it nudges same-domain Ishu ranking upward in this seed, but it does not improve transfer AUC yet.

Next step: run the remaining rows in `reports/assets/combined_v4_transfer_command_manifest.csv` and regenerate this report.
