# Feature Ablation Summary

Run root: `runs\combined_v4_medium_selectk_probe`

## Mean Metrics

| run | feature_set | classifier | select_k | n_runs | accuracy_mean | accuracy_ci_low | accuracy_ci_high | roc_auc_mean | roc_auc_ci_low | roc_auc_ci_high | brier_score_mean | brier_score_ci_low | brier_score_ci_high | expected_calibration_error_mean | expected_calibration_error_ci_low | expected_calibration_error_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| combined_v4_logreg | combined_v4 | logistic_regression | 0 | 3 | 0.7544 | 0.7193 | 0.7982 | 0.8315 | 0.8140 | 0.8544 | 0.1740 | 0.1553 | 0.1842 | 0.1149 | 0.0883 | 0.1311 |
| combined_v4_logreg_selectk60 | combined_v4 | logistic_regression | 60 | 3 | 0.7515 | 0.7281 | 0.7895 | 0.8286 | 0.8165 | 0.8498 | 0.1715 | 0.1547 | 0.1827 | 0.0750 | 0.0522 | 0.0959 |
| combined_v4_logreg_selectk80 | combined_v4 | logistic_regression | 80 | 3 | 0.7310 | 0.6930 | 0.7807 | 0.8269 | 0.7885 | 0.8611 | 0.1794 | 0.1540 | 0.2033 | 0.1014 | 0.0661 | 0.1289 |
| combined_v3_logreg | combined_v3 | logistic_regression | 0 | 3 | 0.7515 | 0.7105 | 0.7982 | 0.8234 | 0.7986 | 0.8393 | 0.1765 | 0.1658 | 0.1909 | 0.1290 | 0.1111 | 0.1442 |

## Top Selected Extra Features

| feature | count | runs |
| --- | --- | --- |
| chroma_laplacian_abs_mean | 6 | combined_v4_logreg_selectk60,combined_v4_logreg_selectk80 |
| chroma_laplacian_entropy | 6 | combined_v4_logreg_selectk60,combined_v4_logreg_selectk80 |
| fft_ring_00_10_ratio | 6 | combined_v4_logreg_selectk60,combined_v4_logreg_selectk80 |
| fft_ring_10_20_ratio | 6 | combined_v4_logreg_selectk60,combined_v4_logreg_selectk80 |
| fft_ring_20_35_ratio | 6 | combined_v4_logreg_selectk60,combined_v4_logreg_selectk80 |
| jpeg_q50_abs_mean | 6 | combined_v4_logreg_selectk60,combined_v4_logreg_selectk80 |
| jpeg_q50_q95_mean_ratio | 6 | combined_v4_logreg_selectk60,combined_v4_logreg_selectk80 |
| recon_half_luma_chroma_ratio | 6 | combined_v4_logreg_selectk60,combined_v4_logreg_selectk80 |
| recon_half_tile16_std_p90 | 6 | combined_v4_logreg_selectk60,combined_v4_logreg_selectk80 |
| recon_quarter_abs_mean | 6 | combined_v4_logreg_selectk60,combined_v4_logreg_selectk80 |
| recon_quarter_luma_chroma_ratio | 6 | combined_v4_logreg_selectk60,combined_v4_logreg_selectk80 |
| fft_high_low_ratio | 5 | combined_v4_logreg_selectk60,combined_v4_logreg_selectk80 |
| fft_ring_35_55_ratio | 5 | combined_v4_logreg_selectk60,combined_v4_logreg_selectk80 |
| fft_spectral_flatness | 5 | combined_v4_logreg_selectk60,combined_v4_logreg_selectk80 |
| recon_quarter_laplacian_abs_mean | 5 | combined_v4_logreg_selectk60,combined_v4_logreg_selectk80 |
