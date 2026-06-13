# combined_v4 Source-Slice Diagnostics

Run date: 2026-06-13

This source-slice diagnostic decomposes the completed `combined_v4` transfer gate by Ishu content category and source-balanced MS COCOAI generator label. It uses the saved `predictions.csv` files, so it is a cheap paper-facing explanation layer rather than a new training run.

## Main Read

`combined_v4_selectk60` helps transfer calibration/ranking on average, but the source-slice view shows why it should stay an ablation: gains are uneven across MS generator sources, and the same model loses accuracy on several Ishu content categories.

## MS COCOAI Select-k60 Detection-Rate Delta

| group_label | n_paired_seeds | fake_detection_rate_delta_mean | accuracy_delta_mean | roc_auc_delta_mean | brier_score_delta_mean | expected_calibration_error_delta_mean |
| --- | --- | --- | --- | --- | --- | --- |
| midjourney6 | 3 | 0.0500 | 0.0500 |  | -0.0446 | -0.0286 |
| dalle3 | 3 | 0.0400 | 0.0400 |  | -0.0233 | -0.0169 |
| sd3 | 3 | 0.0067 | 0.0067 |  | -0.0575 | -0.0349 |
| sdxl | 3 | -0.0033 | -0.0033 |  | 0.0015 | 0.0093 |
| sd21 | 3 | -0.0133 | -0.0133 |  | -0.0310 | -0.0187 |
| real | 3 |  | 0.0013 |  | -0.0002 | 0.0123 |

## Ishu Select-k60 Category Accuracy Delta

| group_label | n_paired_seeds | accuracy_delta_mean | roc_auc_delta_mean | brier_score_delta_mean | expected_calibration_error_delta_mean | predicted_fake_rate_delta_mean |
| --- | --- | --- | --- | --- | --- | --- |
| nature | 3 | 0.0026 | 0.0227 | -0.0134 | -0.0252 | 0.0240 |
| human | 3 | -0.0133 | -0.0200 | 0.0206 | 0.0187 | -0.0170 |
| buildings | 3 | -0.0185 | -0.0089 | 0.0208 | 0.0174 | 0.0185 |
| interior | 3 | -0.0333 | 0.0191 | 0.0056 | 0.0480 | -0.0333 |
| animals | 3 | -0.0521 | 0.0126 | -0.0027 | 0.0700 | 0.0129 |
| items | 3 | -0.0571 | 0.0000 | 0.0177 | 0.0269 | -0.0571 |
| food | 3 | -0.0833 | -0.1356 | 0.0633 | 0.0417 | 0.0833 |

## Raw v4 MS COCOAI Accuracy Delta

| group_label | n_paired_seeds | accuracy_delta_mean | fake_detection_rate_delta_mean | real_false_positive_rate_delta_mean | mean_fake_score_delta_mean |
| --- | --- | --- | --- | --- | --- |
| dalle3 | 3 | 0.0700 | 0.0700 |  | 0.0142 |
| sdxl | 3 | 0.0367 | 0.0367 |  | -0.0032 |
| midjourney6 | 3 | 0.0333 | 0.0333 |  | 0.0130 |
| real | 3 | 0.0067 |  | -0.0067 | -0.0133 |
| sd3 | 3 | -0.0100 | -0.0100 |  | -0.0141 |
| sd21 | 3 | -0.0300 | -0.0300 |  | -0.0236 |

## Mean Metrics By Slice

| phase | group_label | run | n_seeds | accuracy_mean | roc_auc_mean | brier_score_mean | expected_calibration_error_mean | predicted_fake_rate_mean | fake_detection_rate_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ishu_holdout | animals | combined_v3_logreg | 3 | 0.7167 | 0.7870 | 0.1897 | 0.1780 | 0.4460 | 0.6684 |
| ishu_holdout | animals | combined_v4_logreg | 3 | 0.6871 | 0.7834 | 0.2061 | 0.2539 | 0.4164 | 0.6124 |
| ishu_holdout | animals | combined_v4_logreg_selectk60 | 3 | 0.6646 | 0.7997 | 0.1870 | 0.2480 | 0.4589 | 0.6267 |
| ishu_holdout | buildings | combined_v3_logreg | 3 | 0.9327 | 0.9381 | 0.0715 | 0.1227 | 0.7239 | 1.0000 |
| ishu_holdout | buildings | combined_v4_logreg | 3 | 0.9175 | 0.9419 | 0.0697 | 0.1497 | 0.7088 | 0.9762 |
| ishu_holdout | buildings | combined_v4_logreg_selectk60 | 3 | 0.9141 | 0.9292 | 0.0924 | 0.1400 | 0.7424 | 1.0000 |
| ishu_holdout | food | combined_v3_logreg | 3 | 0.8500 | 0.9583 | 0.1136 | 0.2121 | 0.4778 | 0.8889 |
| ishu_holdout | food | combined_v4_logreg | 3 | 0.8167 | 0.9352 | 0.1225 | 0.1672 | 0.5111 | 0.8889 |
| ishu_holdout | food | combined_v4_logreg_selectk60 | 3 | 0.7667 | 0.8228 | 0.1768 | 0.2538 | 0.5611 | 0.8889 |
| ishu_holdout | human | combined_v3_logreg | 3 | 0.8357 | 0.8924 | 0.1297 | 0.1901 | 0.4997 | 0.9667 |
| ishu_holdout | human | combined_v4_logreg | 3 | 0.8357 | 0.8904 | 0.1313 | 0.1708 | 0.4997 | 0.9667 |
| ishu_holdout | human | combined_v4_logreg_selectk60 | 3 | 0.8224 | 0.8724 | 0.1503 | 0.2088 | 0.4827 | 0.9111 |
| ishu_holdout | interior | combined_v3_logreg | 3 | 0.8583 | 0.9392 | 0.1295 | 0.2052 | 0.5750 | 0.8472 |
| ishu_holdout | interior | combined_v4_logreg | 3 | 0.8250 | 0.9566 | 0.1257 | 0.2188 | 0.5417 | 0.8056 |
| ishu_holdout | interior | combined_v4_logreg_selectk60 | 3 | 0.8250 | 0.9583 | 0.1352 | 0.2532 | 0.5417 | 0.8056 |
| ishu_holdout | items | combined_v3_logreg | 3 | 0.9190 | 1.0000 | 0.0501 | 0.1208 | 0.3286 | 0.7778 |
| ishu_holdout | items | combined_v4_logreg | 3 | 0.9190 | 1.0000 | 0.0438 | 0.1131 | 0.3286 | 0.7778 |
| ishu_holdout | items | combined_v4_logreg_selectk60 | 3 | 0.8619 | 1.0000 | 0.0679 | 0.1478 | 0.2714 | 0.6111 |
| ishu_holdout | nature | combined_v3_logreg | 3 | 0.7135 | 0.8239 | 0.1944 | 0.2438 | 0.4772 | 0.7010 |
| ishu_holdout | nature | combined_v4_logreg | 3 | 0.7321 | 0.8493 | 0.1827 | 0.2471 | 0.4854 | 0.7380 |
| ishu_holdout | nature | combined_v4_logreg_selectk60 | 3 | 0.7162 | 0.8466 | 0.1810 | 0.2186 | 0.5013 | 0.7380 |
| ishu_to_ms_cocoai | dalle3 | combined_v3_logreg | 3 | 0.1900 |  | 0.5779 | 0.7173 | 0.1900 | 0.1900 |
| ishu_to_ms_cocoai | dalle3 | combined_v4_logreg | 3 | 0.2600 |  | 0.5705 | 0.7031 | 0.2600 | 0.2600 |
| ishu_to_ms_cocoai | dalle3 | combined_v4_logreg_selectk60 | 3 | 0.2300 |  | 0.5546 | 0.7004 | 0.2300 | 0.2300 |
| ishu_to_ms_cocoai | midjourney6 | combined_v3_logreg | 3 | 0.2467 |  | 0.5316 | 0.6707 | 0.2467 | 0.2467 |
| ishu_to_ms_cocoai | midjourney6 | combined_v4_logreg | 3 | 0.2800 |  | 0.5275 | 0.6578 | 0.2800 | 0.2800 |
| ishu_to_ms_cocoai | midjourney6 | combined_v4_logreg_selectk60 | 3 | 0.2967 |  | 0.4870 | 0.6421 | 0.2967 | 0.2967 |
| ishu_to_ms_cocoai | real | combined_v3_logreg | 3 | 0.8727 |  | 0.0964 | 0.2063 | 0.1273 |  |
| ishu_to_ms_cocoai | real | combined_v4_logreg | 3 | 0.8793 |  | 0.0913 | 0.1930 | 0.1207 |  |
| ishu_to_ms_cocoai | real | combined_v4_logreg_selectk60 | 3 | 0.8740 |  | 0.0962 | 0.2186 | 0.1260 |  |
| ishu_to_ms_cocoai | sd21 | combined_v3_logreg | 3 | 0.1300 |  | 0.7111 | 0.8123 | 0.1300 | 0.1300 |
| ishu_to_ms_cocoai | sd21 | combined_v4_logreg | 3 | 0.1000 |  | 0.7547 | 0.8359 | 0.1000 | 0.1000 |
| ishu_to_ms_cocoai | sd21 | combined_v4_logreg_selectk60 | 3 | 0.1167 |  | 0.6801 | 0.7936 | 0.1167 | 0.1167 |
| ishu_to_ms_cocoai | sd3 | combined_v3_logreg | 3 | 0.0633 |  | 0.7633 | 0.8554 | 0.0633 | 0.0633 |
| ishu_to_ms_cocoai | sd3 | combined_v4_logreg | 3 | 0.0533 |  | 0.7858 | 0.8695 | 0.0533 | 0.0533 |
| ishu_to_ms_cocoai | sd3 | combined_v4_logreg_selectk60 | 3 | 0.0700 |  | 0.7058 | 0.8205 | 0.0700 | 0.0700 |
| ishu_to_ms_cocoai | sdxl | combined_v3_logreg | 3 | 0.4733 |  | 0.3509 | 0.5172 | 0.4733 | 0.4733 |
| ishu_to_ms_cocoai | sdxl | combined_v4_logreg | 3 | 0.5100 |  | 0.3644 | 0.5204 | 0.5100 | 0.5100 |
| ishu_to_ms_cocoai | sdxl | combined_v4_logreg_selectk60 | 3 | 0.4700 |  | 0.3524 | 0.5265 | 0.4700 | 0.4700 |
