# Feature Ablation Summary

Run root: `runs\combined_v4_selectk_probe`

## Mean Metrics

| run | feature_set | classifier | select_k | n_runs | accuracy_mean | accuracy_ci_low | accuracy_ci_high | roc_auc_mean | roc_auc_ci_low | roc_auc_ci_high | brier_score_mean | brier_score_ci_low | brier_score_ci_high | expected_calibration_error_mean | expected_calibration_error_ci_low | expected_calibration_error_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| combined_v4_logreg_selectk60 | combined_v4 | logistic_regression | 60 | 3 | 0.7389 | 0.7333 | 0.7500 | 0.8219 | 0.8089 | 0.8300 | 0.1744 | 0.1675 | 0.1779 | 0.1448 | 0.1396 | 0.1541 |
| combined_v4_logreg_selectk80 | combined_v4 | logistic_regression | 80 | 3 | 0.7611 | 0.7500 | 0.7667 | 0.8041 | 0.7800 | 0.8178 | 0.1872 | 0.1801 | 0.1974 | 0.1560 | 0.1295 | 0.1869 |
| combined_v3_logreg | combined_v3 | logistic_regression | 0 | 3 | 0.7278 | 0.7000 | 0.7667 | 0.8033 | 0.7511 | 0.8300 | 0.1876 | 0.1652 | 0.2176 | 0.1451 | 0.1400 | 0.1528 |
| combined_v4_logreg | combined_v4 | logistic_regression | 0 | 3 | 0.7167 | 0.6500 | 0.7667 | 0.8007 | 0.7822 | 0.8267 | 0.1881 | 0.1759 | 0.2044 | 0.1554 | 0.1354 | 0.1831 |
| combined_v4_logreg_selectk40 | combined_v4 | logistic_regression | 40 | 3 | 0.7222 | 0.6833 | 0.7667 | 0.7737 | 0.7256 | 0.8367 | 0.1960 | 0.1642 | 0.2178 | 0.1247 | 0.0891 | 0.1631 |
| combined_v4_hgb_selectk60 | combined_v4 | hist_gradient_boosting | 60 | 3 | 0.6833 | 0.6333 | 0.7833 | 0.7556 | 0.6800 | 0.8767 | 0.2289 | 0.1491 | 0.2870 | 0.2135 | 0.1327 | 0.2648 |
| combined_v3_hgb | combined_v3 | hist_gradient_boosting | 0 | 3 | 0.6667 | 0.6167 | 0.7500 | 0.7389 | 0.6511 | 0.8400 | 0.2383 | 0.1738 | 0.2957 | 0.2105 | 0.1532 | 0.2822 |
| combined_v4_hgb | combined_v4 | hist_gradient_boosting | 0 | 3 | 0.6500 | 0.5833 | 0.7500 | 0.7141 | 0.6478 | 0.8278 | 0.2481 | 0.1816 | 0.2860 | 0.2140 | 0.1517 | 0.2609 |

## Top Selected Extra Features

| feature | count | runs |
| --- | --- | --- |
| fft_ring_10_20_ratio | 12 | combined_v4_hgb_selectk60,combined_v4_logreg_selectk40,combined_v4_logreg_selectk60,combined_v4_logreg_selectk80 |
| recon_half_luma_chroma_ratio | 12 | combined_v4_hgb_selectk60,combined_v4_logreg_selectk40,combined_v4_logreg_selectk60,combined_v4_logreg_selectk80 |
| recon_quarter_luma_chroma_ratio | 12 | combined_v4_hgb_selectk60,combined_v4_logreg_selectk40,combined_v4_logreg_selectk60,combined_v4_logreg_selectk80 |
| fft_ring_00_10_ratio | 11 | combined_v4_hgb_selectk60,combined_v4_logreg_selectk40,combined_v4_logreg_selectk60,combined_v4_logreg_selectk80 |
| fft_ring_20_35_ratio | 11 | combined_v4_hgb_selectk60,combined_v4_logreg_selectk40,combined_v4_logreg_selectk60,combined_v4_logreg_selectk80 |
| chroma_laplacian_entropy | 9 | combined_v4_hgb_selectk60,combined_v4_logreg_selectk40,combined_v4_logreg_selectk60,combined_v4_logreg_selectk80 |
| chroma_laplacian_abs_mean | 8 | combined_v4_hgb_selectk60,combined_v4_logreg_selectk40,combined_v4_logreg_selectk60,combined_v4_logreg_selectk80 |
| jpeg_q50_q95_mean_ratio | 8 | combined_v4_hgb_selectk60,combined_v4_logreg_selectk40,combined_v4_logreg_selectk60,combined_v4_logreg_selectk80 |
| fft_spectral_flatness | 7 | combined_v4_hgb_selectk60,combined_v4_logreg_selectk60,combined_v4_logreg_selectk80 |
| recon_half_tile16_std_p90 | 7 | combined_v4_hgb_selectk60,combined_v4_logreg_selectk40,combined_v4_logreg_selectk60,combined_v4_logreg_selectk80 |
| fft_ring_55_100_ratio | 6 | combined_v4_hgb_selectk60,combined_v4_logreg_selectk60,combined_v4_logreg_selectk80 |
| chroma_edge_corr | 5 | combined_v4_hgb_selectk60,combined_v4_logreg_selectk40,combined_v4_logreg_selectk60,combined_v4_logreg_selectk80 |
| fft_high_low_ratio | 5 | combined_v4_hgb_selectk60,combined_v4_logreg_selectk40,combined_v4_logreg_selectk60,combined_v4_logreg_selectk80 |
| jpeg_q50_abs_mean | 5 | combined_v4_hgb_selectk60,combined_v4_logreg_selectk40,combined_v4_logreg_selectk60,combined_v4_logreg_selectk80 |
| recon_half_abs_p95 | 5 | combined_v4_hgb_selectk60,combined_v4_logreg_selectk40,combined_v4_logreg_selectk60,combined_v4_logreg_selectk80 |
