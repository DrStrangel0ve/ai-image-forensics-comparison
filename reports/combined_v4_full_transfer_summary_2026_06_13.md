# combined_v4 Full-Transfer Summary

Run date: 2026-06-13

This report summarizes the checked local `metrics.json` files for the `combined_v4` transfer gate: Ishu same-domain holdout plus Ishu -> source-balanced MS COCOAI transfer.

Seeds included: 7, 17, 29. This is now a repeated-seed transfer summary.

## Seed Results

| phase_label | seed | run | accuracy | roc_auc | brier_score | expected_calibration_error | fake_call_rate | real_false_positive_rate | fake_miss_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ishu holdout split | 7 | combined_v3_logreg | 0.8158 | 0.8938 | 0.1324 | 0.0807 | 0.5000 | 0.1897 | 0.1786 |
| Ishu holdout split | 7 | combined_v4_logreg | 0.8070 | 0.8990 | 0.1340 | 0.0752 | 0.4737 | 0.1724 | 0.2143 |
| Ishu holdout split | 7 | combined_v4_logreg_selectk60 | 0.7982 | 0.8867 | 0.1373 | 0.0740 | 0.4825 | 0.1897 | 0.2143 |
| Ishu holdout split | 17 | combined_v3_logreg | 0.8246 | 0.9089 | 0.1229 | 0.0472 | 0.5263 | 0.2069 | 0.1429 |
| Ishu holdout split | 17 | combined_v4_logreg | 0.7982 | 0.9089 | 0.1248 | 0.0763 | 0.5175 | 0.2241 | 0.1786 |
| Ishu holdout split | 17 | combined_v4_logreg_selectk60 | 0.7895 | 0.8784 | 0.1399 | 0.0783 | 0.5439 | 0.2586 | 0.1607 |
| Ishu holdout split | 29 | combined_v3_logreg | 0.8333 | 0.8799 | 0.1360 | 0.1011 | 0.5175 | 0.1897 | 0.1429 |
| Ishu holdout split | 29 | combined_v4_logreg | 0.8333 | 0.8830 | 0.1333 | 0.1052 | 0.5351 | 0.2069 | 0.1250 |
| Ishu holdout split | 29 | combined_v4_logreg_selectk60 | 0.7982 | 0.8670 | 0.1487 | 0.0744 | 0.5351 | 0.2414 | 0.1607 |
| Ishu -> source-balanced MS COCOAI | 7 | combined_v3_logreg | 0.5430 | 0.5750 | 0.3435 | 0.2969 | 0.1790 | 0.1360 | 0.7780 |
| Ishu -> source-balanced MS COCOAI | 7 | combined_v4_logreg | 0.5520 | 0.5685 | 0.3483 | 0.2910 | 0.1960 | 0.1440 | 0.7520 |
| Ishu -> source-balanced MS COCOAI | 7 | combined_v4_logreg_selectk60 | 0.5590 | 0.5976 | 0.3154 | 0.2470 | 0.2050 | 0.1460 | 0.7360 |
| Ishu -> source-balanced MS COCOAI | 17 | combined_v3_logreg | 0.5540 | 0.5818 | 0.3385 | 0.2842 | 0.1840 | 0.1300 | 0.7620 |
| Ishu -> source-balanced MS COCOAI | 17 | combined_v4_logreg | 0.5670 | 0.5837 | 0.3430 | 0.2942 | 0.1890 | 0.1220 | 0.7440 |
| Ishu -> source-balanced MS COCOAI | 17 | combined_v4_logreg_selectk60 | 0.5640 | 0.6183 | 0.3179 | 0.2657 | 0.1720 | 0.1080 | 0.7640 |
| Ishu -> source-balanced MS COCOAI | 29 | combined_v3_logreg | 0.5430 | 0.5840 | 0.3431 | 0.2923 | 0.1590 | 0.1160 | 0.7980 |
| Ishu -> source-balanced MS COCOAI | 29 | combined_v4_logreg | 0.5610 | 0.5879 | 0.3465 | 0.3041 | 0.1570 | 0.0960 | 0.7820 |
| Ishu -> source-balanced MS COCOAI | 29 | combined_v4_logreg_selectk60 | 0.5430 | 0.5606 | 0.3450 | 0.2861 | 0.1670 | 0.1240 | 0.7900 |

## Mean Results

| phase_label | run | n_seeds | accuracy_mean | roc_auc_mean | brier_score_mean | expected_calibration_error_mean | fake_call_rate_mean | roc_auc_ci_low | roc_auc_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ishu holdout split | combined_v3_logreg | 3 | 0.8246 | 0.8942 | 0.1304 | 0.0764 | 0.5146 | 0.8799 | 0.9089 |
| Ishu holdout split | combined_v4_logreg | 3 | 0.8129 | 0.8970 | 0.1307 | 0.0855 | 0.5088 | 0.8830 | 0.9089 |
| Ishu holdout split | combined_v4_logreg_selectk60 | 3 | 0.7953 | 0.8774 | 0.1420 | 0.0756 | 0.5205 | 0.8670 | 0.8867 |
| Ishu -> source-balanced MS COCOAI | combined_v3_logreg | 3 | 0.5467 | 0.5803 | 0.3417 | 0.2911 | 0.1740 | 0.5750 | 0.5840 |
| Ishu -> source-balanced MS COCOAI | combined_v4_logreg | 3 | 0.5600 | 0.5801 | 0.3459 | 0.2964 | 0.1807 | 0.5685 | 0.5879 |
| Ishu -> source-balanced MS COCOAI | combined_v4_logreg_selectk60 | 3 | 0.5553 | 0.5922 | 0.3261 | 0.2663 | 0.1813 | 0.5606 | 0.6183 |

## Paired Delta Versus combined_v3

| phase_label | candidate | n_paired_seeds | accuracy_delta_mean | roc_auc_delta_mean | brier_score_delta_mean | expected_calibration_error_delta_mean | fake_call_rate_delta_mean | roc_auc_delta_ci_low | roc_auc_delta_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ishu holdout split | combined_v4_logreg | 3 | -0.0117 | 0.0028 | 0.0003 | 0.0092 | -0.0058 | 0.0000 | 0.0052 |
| Ishu holdout split | combined_v4_logreg_selectk60 | 3 | -0.0292 | -0.0168 | 0.0115 | -0.0008 | 0.0058 | -0.0305 | -0.0071 |
| Ishu -> source-balanced MS COCOAI | combined_v4_logreg | 3 | 0.0133 | -0.0002 | 0.0042 | 0.0053 | 0.0067 | -0.0065 | 0.0039 |
| Ishu -> source-balanced MS COCOAI | combined_v4_logreg_selectk60 | 3 | 0.0087 | 0.0119 | -0.0156 | -0.0249 | 0.0073 | -0.0233 | 0.0365 |

## Interpretation

The full three-seed gate does not justify promoting `combined_v4` as the main conventional branch. Raw v4 improves Ishu -> MS COCOAI transfer accuracy by +0.0133, but its transfer AUC is effectively flat at -0.0002 and its Brier/ECE move by +0.0042 / +0.0053.

`combined_v4_selectk60` is the more useful ablation: it improves transfer AUC by +0.0119 and Brier/ECE by -0.0156 / -0.0249, with a smaller transfer accuracy gain of +0.0087. The cost is same-domain Ishu degradation: -0.0292 accuracy and -0.0168 AUC.

Decision: keep `combined_v3` as the main conventional baseline for now, and use `combined_v4_selectk60` as a calibration/transfer ablation in the WIFS/DFF appendix. The next useful v4 experiment is not another same split rerun; it is source-aware feature selection or a stronger regularized classifier.
