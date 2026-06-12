# Combined V4 Medium Select-K Probe

Run date: 2026-06-12

This probe reruns the `combined_v4` ablation at a larger, still-bounded scale using the reusable grid runner:

- dataset: Ishu AI-vs-real 2026;
- seeds: 7, 17, 29;
- train cap: 240 images per seed;
- validation cap: 120 requested, 114 available after the deterministic split;
- image size: 64;
- classifier: balanced logistic regression.

The goal is to test whether the earlier small-split result was mostly a dimensionality problem. The answer is yes: with more training images, raw `combined_v4` becomes the strongest mean AUC and mean accuracy setting.

## Mean Results

Generated assets are checked into `reports/assets/combined_v4_medium_selectk_probe/`.
The regenerated summary CSV/report now includes 95% deterministic bootstrap confidence intervals across the three seeds.

| method | mean accuracy | mean ROC AUC | mean Brier | mean ECE |
| --- | ---: | ---: | ---: | ---: |
| raw `combined_v4` logistic | 0.7544 | 0.8315 | 0.1740 | 0.1149 |
| `combined_v4` logistic select-k60 | 0.7515 | 0.8286 | 0.1715 | 0.0750 |
| `combined_v4` logistic select-k80 | 0.7310 | 0.8269 | 0.1794 | 0.1014 |
| `combined_v3` logistic | 0.7515 | 0.8234 | 0.1765 | 0.1290 |

## Interpretation

This medium run changes the v4 story:

- Raw `combined_v4` now has the best mean accuracy and AUC, narrowly beating `combined_v3`.
- The interval-aware asset makes that a narrow ablation lead, not a final significance claim: raw `combined_v4` AUC is `0.8315` with a 95% bootstrap interval of `[0.8140, 0.8544]`, while `combined_v3` AUC is `0.8234` with interval `[0.7986, 0.8393]`.
- `combined_v4` select-k60 has the best Brier score and ECE, so feature selection remains useful as a calibration-friendly variant.
- `combined_v4` select-k80 keeps strong AUC but has lower default-threshold accuracy on these splits.
- The recurrent v4-only selected features are not random: chroma Laplacian stats, low/mid FFT rings, JPEG q50 probes, and reconstruction luma/chroma ratios appear in every selected run.

For SCP-Fusion v1, this suggests carrying two conventional candidates into the next stage:

- raw `combined_v4` for ranking strength;
- `combined_v4` select-k60 for calibration-sensitive score fusion.

The next fair test is a larger or full repeated-seed run on Ishu, followed by MS COCOAI and source-heldout transfer. If raw v4 continues to lead with more data, use it as the primary conventional branch and keep select-k60 as a calibration ablation.

## Reproduction

```powershell
.\.venv\Scripts\python scripts\run_feature_ablation_grid.py `
  --data-dir data\raw\ishu_ai_vs_real_2026 `
  --out-dir runs\combined_v4_medium_selectk_probe `
  --summary-dir reports\assets\combined_v4_medium_selectk_probe `
  --seeds 7 17 29 `
  --image-size 64 `
  --max-train-samples 240 `
  --max-test-samples 120 `
  --skip-errors `
  --skip-existing
```
