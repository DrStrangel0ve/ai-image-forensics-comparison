# Combined V4 Probe

Run date: 2026-06-12

This probe adds `noise_v4` and `combined_v4`, the first checked-in version of the reconstruction, multiscale frequency, chroma, and JPEG expansion proposed in the SCP-Fusion roadmap. It keeps all `combined_v3` features and appends 27 new noise/signal features, so saved `combined_v3` models remain backward-compatible.

## Feature Additions

`combined_v4` adds:

- bicubic half-scale and quarter-scale reconstruction residual summaries as an AEROBLADE-lite proxy;
- multiring FFT energy ratios, spectral flatness, radial slope, and high/low frequency ratio;
- chroma-edge correlation, chroma Laplacian entropy, and chroma block-boundary ratios;
- stronger JPEG recompression sensitivity at quality 50 and a quality-70 versus quality-95 phase-8 contrast delta.

Feature counts:

| feature set | feature count |
| --- | ---: |
| `combined_v3` | 76 |
| `combined_v4` | 103 |

## Bounded Ishu Smoke Probe

This was intentionally small: 80 train images, 40 validation images, image size 64, seed 7, logistic regression. It checks that the new features are finite, trainable, and comparable to `combined_v3`; it should not be treated as a full benchmark.

| feature set | accuracy | ROC AUC | Brier | ECE |
| --- | ---: | ---: | ---: | ---: |
| `combined_v3` | 0.6750 | 0.7725 | 0.1922 | 0.1762 |
| `combined_v4` | 0.6500 | 0.7425 | 0.2102 | 0.2031 |

## Interpretation

The v4 feature family is implemented and usable, but the first tiny logistic-regression probe does not beat `combined_v3`. That is still useful evidence: adding reconstruction and extra spectral/chroma/JPEG scalars increases forensic coverage, but it also increases dimensionality and may need stronger regularization, feature selection, tree-based models, or full repeated-seed training before it helps SCP-Fusion.

For the paper roadmap, this means `combined_v4` should be described as an ablation candidate, not as an improvement yet. The next fair comparison is a repeated-seed `combined_v3` versus `combined_v4` run on Ishu and MS COCOAI, ideally with both logistic regression and histogram gradient boosting.

## Reproduction

```powershell
.\.venv\Scripts\python scripts\run_feature_baseline.py `
  --data-dir data\raw\ishu_ai_vs_real_2026 `
  --output-dir runs\combined_v4_probe\ishu_limited_combined_v3 `
  --feature-set combined_v3 `
  --classifier logistic_regression `
  --image-size 64 `
  --seed 7 `
  --max-train-samples 80 `
  --max-test-samples 40 `
  --skip-errors
```

```powershell
.\.venv\Scripts\python scripts\run_feature_baseline.py `
  --data-dir data\raw\ishu_ai_vs_real_2026 `
  --output-dir runs\combined_v4_probe\ishu_limited_combined_v4 `
  --feature-set combined_v4 `
  --classifier logistic_regression `
  --image-size 64 `
  --seed 7 `
  --max-train-samples 80 `
  --max-test-samples 40 `
  --skip-errors
```
