# Feature Ablation Summary

Run root: `runs\combined_v4_medium_selectk_probe`

## Mean Metrics

| run | feature_set | classifier | select_k | n_runs | accuracy_mean | roc_auc_mean | brier_score_mean | expected_calibration_error_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| combined_v4_logreg | combined_v4 | logistic_regression | 0 | 3 | 0.7544 | 0.8315 | 0.1740 | 0.1149 |
| combined_v4_logreg_selectk60 | combined_v4 | logistic_regression | 60 | 3 | 0.7515 | 0.8286 | 0.1715 | 0.0750 |
| combined_v4_logreg_selectk80 | combined_v4 | logistic_regression | 80 | 3 | 0.7310 | 0.8269 | 0.1794 | 0.1014 |
| combined_v3_logreg | combined_v3 | logistic_regression | 0 | 3 | 0.7515 | 0.8234 | 0.1765 | 0.1290 |

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
