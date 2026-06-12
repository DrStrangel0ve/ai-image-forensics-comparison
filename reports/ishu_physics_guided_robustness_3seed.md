# Ishu Physics-Guided Robustness, Three Seeds

Run date: 2026-06-12

This extends the seed-29 robustness check to all three deterministic Ishu splits used in the main benchmark: seeds 7, 17, and 29. For each seed, the exact held-out test split is exported with `scripts/export_image_split.py`, transformed with `scripts/make_robustness_variants.py`, and evaluated against the saved `combined_v3`, ResNet-18, and physics-guided fusion models from that seed.

Each seed contributes 114 clean test images. Each robustness variant therefore has 342 total evaluations per method across the three seeds.

## Overall Means

Mean over 12 checks: 3 seeds x 4 transforms.

| method | n_checks | accuracy_mean | accuracy_delta_mean | roc_auc_mean | roc_auc_delta_mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| `combined_v3` | 12 | 0.7924 | -0.0322 | 0.8751 | -0.0191 |
| ResNet-18 | 12 | 0.8231 | -0.0015 | 0.8904 | -0.0022 |
| physics-guided ResNet-18 + `combined_v3` | 12 | 0.8443 | -0.0007 | 0.9189 | 0.0012 |

Win counts across the same 12 seed-plus-transform checks:

| metric | `combined_v3` wins | ResNet-18 wins | physics-guided fusion wins |
| --- | ---: | ---: | ---: |
| accuracy | 1 | 1 | 10 |
| roc_auc | 2 | 0 | 10 |

## Variant Means

Mean over seeds 7, 17, and 29.

| variant | method | accuracy_mean | accuracy_delta_mean | roc_auc_mean | roc_auc_delta_mean |
| --- | --- | ---: | ---: | ---: | ---: |
| blur1 | `combined_v3` | 0.7719 | -0.0526 | 0.8586 | -0.0356 |
| blur1 | ResNet-18 | 0.8304 | 0.0058 | 0.8968 | 0.0041 |
| blur1 | physics-guided fusion | 0.8567 | 0.0117 | 0.9274 | 0.0097 |
| crop85 | `combined_v3` | 0.8129 | -0.0117 | 0.8912 | -0.0030 |
| crop85 | ResNet-18 | 0.8012 | -0.0234 | 0.8748 | -0.0179 |
| crop85 | physics-guided fusion | 0.8304 | -0.0146 | 0.9050 | -0.0127 |
| jpeg70 | `combined_v3` | 0.8099 | -0.0146 | 0.8873 | -0.0069 |
| jpeg70 | ResNet-18 | 0.8333 | 0.0088 | 0.8940 | 0.0013 |
| jpeg70 | physics-guided fusion | 0.8421 | -0.0029 | 0.9187 | 0.0010 |
| resize_half | `combined_v3` | 0.7749 | -0.0497 | 0.8631 | -0.0311 |
| resize_half | ResNet-18 | 0.8275 | 0.0029 | 0.8962 | 0.0036 |
| resize_half | physics-guided fusion | 0.8480 | 0.0029 | 0.9246 | 0.0069 |

## Interpretation

The three-seed result confirms the seed-29 finding. Physics-guided fusion is not merely benefiting from one lucky split; it is the most robust model family across the tested transforms.

The main weakness of standalone `combined_v3` is still blur and half-resolution resize. Its mean accuracy drops by 5.3 points under blur and 5.0 points under resize. ResNet-18 is much steadier, but the fused model preserves nearly all of ResNet's transform stability while keeping the extra forensic-feature lift.

Crop is the hardest transform for the fused model by AUC, with a mean AUC drop of 0.0127. Even there, the fused model still has the highest mean accuracy and AUC among the three methods.

This remains an in-dataset robustness test. The next fair stress test is cross-domain robustness: train on Ishu or MS COCOAI, transform the other dataset, and evaluate without target tuning.

## Reproduce

Export the held-out split for each seed:

```powershell
python scripts/export_image_split.py `
  --data-dir data/raw/ishu_ai_vs_real_2026 `
  --out-dir data/raw/ishu_ai_vs_real_2026_seed7_test `
  --split test `
  --seed 7 `
  --val-fraction 0.2
```

Repeat with `--seed 17` and `--seed 29`, changing the output folder name.

Create variants for each exported test folder:

```powershell
python scripts/make_robustness_variants.py `
  --data-dir data/raw/ishu_ai_vs_real_2026_seed7_test `
  --out-dir data/raw/ishu_ai_vs_real_2026_seed7_test_robustness_variants `
  --variants jpeg70 blur1 resize_half crop85 `
  --format jpg
```

Evaluate each variant with:

- `scripts/evaluate_feature_model.py`
- `scripts/evaluate_neural_net.py`
- `scripts/evaluate_physics_guided_net.py`

Then summarize each seed with `scripts/summarize_robustness_eval.py`. The three seed summaries used here are:

- `runs/ishu_ai_vs_real_2026_seed7_robustness/summary/robustness_summary.csv`
- `runs/ishu_ai_vs_real_2026_seed17_robustness/summary/robustness_summary.csv`
- `runs/ishu_ai_vs_real_2026_seed29_robustness/summary/robustness_summary.csv`
