# Combined V4 Feature-Selection Probe

Run date: 2026-06-12

This follow-up tests whether the larger `combined_v4` feature family needs regularization rather than being discarded. The runner now supports `--select-k`, which wraps the conventional classifier with univariate feature selection and records the selected feature names in `metrics.json`.

This probe is still bounded: Ishu AI-vs-real, seeds 7/17/29, 120 train images, 60 validation images, image size 64, balanced logistic regression. It is meant to choose the next full ablation, not to replace the existing full-size Ishu/MS COCOAI benchmarks.

## Three-Seed Result

| seed | method | selected features | accuracy | ROC AUC | Brier | ECE |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 7 | `combined_v3` logistic | 0 | 0.7000 | 0.8300 | 0.1652 | 0.1528 |
| 7 | `combined_v4` logistic | 0 | 0.6500 | 0.7933 | 0.1842 | 0.1831 |
| 7 | `combined_v4` logistic select-k | 60 | 0.7333 | 0.8089 | 0.1778 | 0.1541 |
| 17 | `combined_v3` logistic | 0 | 0.7667 | 0.8289 | 0.1801 | 0.1400 |
| 17 | `combined_v4` logistic | 0 | 0.7667 | 0.8267 | 0.1759 | 0.1354 |
| 17 | `combined_v4` logistic select-k | 60 | 0.7500 | 0.8267 | 0.1779 | 0.1408 |
| 29 | `combined_v3` logistic | 0 | 0.7167 | 0.7511 | 0.2176 | 0.1423 |
| 29 | `combined_v4` logistic | 0 | 0.7333 | 0.7822 | 0.2044 | 0.1476 |
| 29 | `combined_v4` logistic select-k | 60 | 0.7333 | 0.8300 | 0.1675 | 0.1396 |

Mean metrics:

| method | mean accuracy | mean ROC AUC | mean Brier | mean ECE |
| --- | ---: | ---: | ---: | ---: |
| `combined_v4` logistic select-k | 0.7389 | 0.8219 | 0.1744 | 0.1448 |
| `combined_v3` logistic | 0.7278 | 0.8033 | 0.1876 | 0.1451 |
| `combined_v4` logistic | 0.7167 | 0.8007 | 0.1881 | 0.1554 |

## Selected V4 Signals

Across the three select-k runs, 40 of the 180 selected feature slots were new `combined_v4` features. These v4-only features were selected in all three seeds:

- `recon_half_luma_chroma_ratio`
- `recon_quarter_luma_chroma_ratio`
- `fft_ring_00_10_ratio`
- `fft_ring_10_20_ratio`
- `fft_ring_20_35_ratio`

This is the first evidence that the new reconstruction and spectral features are not just dead weight. Raw `combined_v4` still underperforms slightly, but selected `combined_v4` has the best mean AUC and Brier score in this bounded probe.

## Expanded K and Tree Sweep

A follow-up sweep adds `k=40`, `k=80`, and histogram-gradient-boosting variants. The generated CSVs and selected-feature frequencies are checked into `reports/assets/combined_v4_selectk_probe/`.

| method | mean accuracy | mean ROC AUC | mean Brier | mean ECE |
| --- | ---: | ---: | ---: | ---: |
| `combined_v4` logistic select-k60 | 0.7389 | 0.8219 | 0.1744 | 0.1448 |
| `combined_v4` logistic select-k80 | 0.7611 | 0.8041 | 0.1872 | 0.1560 |
| `combined_v3` logistic | 0.7278 | 0.8033 | 0.1876 | 0.1451 |
| raw `combined_v4` logistic | 0.7167 | 0.8007 | 0.1881 | 0.1554 |
| `combined_v4` logistic select-k40 | 0.7222 | 0.7737 | 0.1960 | 0.1247 |
| `combined_v4` HGB select-k60 | 0.6833 | 0.7556 | 0.2289 | 0.2135 |
| `combined_v3` HGB | 0.6667 | 0.7389 | 0.2383 | 0.2105 |
| raw `combined_v4` HGB | 0.6500 | 0.7141 | 0.2481 | 0.2140 |

The expanded sweep keeps the same story but makes it sharper:

- `k=60` is the best ranking setting by AUC and Brier score.
- `k=80` is the best default-threshold accuracy setting, but has worse AUC/calibration.
- histogram gradient boosting underperforms logistic regression on this small, high-dimensional split, so the next larger run should prioritize selected logistic regression before spending GPU/CPU time on tree models.
- the same v4-only signals recur across selected runs: `recon_half_luma_chroma_ratio`, `recon_quarter_luma_chroma_ratio`, and low/mid FFT ring ratios.

## Interpretation

For SCP-Fusion, `combined_v4` should move forward as a tuned branch, not as a raw feature dump. The next full ablation should compare:

- `combined_v3` logistic regression;
- raw `combined_v4` logistic regression;
- selected `combined_v4` logistic regression with `k=60` and `k=80`;
- tree models only if a larger split changes the bounded result.

Run that on full or larger repeated Ishu splits first, then repeat the winner on MS COCOAI and source-heldout transfer.

## Reproduction

```powershell
$configs = @(
  @{Name='combined_v3_logreg'; Feature='combined_v3'; SelectK=0},
  @{Name='combined_v4_logreg'; Feature='combined_v4'; SelectK=0},
  @{Name='combined_v4_logreg_selectk60'; Feature='combined_v4'; SelectK=60}
)
foreach ($seed in @(7,17,29)) {
  foreach ($config in $configs) {
    .\.venv\Scripts\python scripts\run_feature_baseline.py `
      --data-dir data\raw\ishu_ai_vs_real_2026 `
      --output-dir "runs\combined_v4_selectk_probe\seed$seed\$($config.Name)" `
      --feature-set $config.Feature `
      --classifier logistic_regression `
      --select-k $config.SelectK `
      --image-size 64 `
      --seed $seed `
      --max-train-samples 120 `
      --max-test-samples 60 `
      --skip-errors
  }
}
```

Summarize the sweep:

```powershell
.\.venv\Scripts\python scripts\summarize_feature_ablation.py `
  --run-root runs\combined_v4_selectk_probe `
  --out-dir reports\assets\combined_v4_selectk_probe `
  --extra-feature-base combined_v3
```
