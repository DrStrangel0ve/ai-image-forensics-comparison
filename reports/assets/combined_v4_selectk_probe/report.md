# Feature Ablation Summary

Run root: `runs\combined_v4_selectk_probe`

## Mean Metrics

| run | feature_set | classifier | select_k | n_runs | accuracy_mean | roc_auc_mean | brier_score_mean | expected_calibration_error_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| combined_v4_logreg_selectk60 | combined_v4 | logistic_regression | 60 | 3 | 0.7389 | 0.8219 | 0.1744 | 0.1448 |
| combined_v4_logreg_selectk80 | combined_v4 | logistic_regression | 80 | 3 | 0.7611 | 0.8041 | 0.1872 | 0.1560 |
| combined_v3_logreg | combined_v3 | logistic_regression | 0 | 3 | 0.7278 | 0.8033 | 0.1876 | 0.1451 |
| combined_v4_logreg | combined_v4 | logistic_regression | 0 | 3 | 0.7167 | 0.8007 | 0.1881 | 0.1554 |
| combined_v4_logreg_selectk40 | combined_v4 | logistic_regression | 40 | 3 | 0.7222 | 0.7737 | 0.1960 | 0.1247 |
| combined_v4_hgb_selectk60 | combined_v4 | hist_gradient_boosting | 60 | 3 | 0.6833 | 0.7556 | 0.2289 | 0.2135 |
| combined_v3_hgb | combined_v3 | hist_gradient_boosting | 0 | 3 | 0.6667 | 0.7389 | 0.2383 | 0.2105 |
| combined_v4_hgb | combined_v4 | hist_gradient_boosting | 0 | 3 | 0.6500 | 0.7141 | 0.2481 | 0.2140 |

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
