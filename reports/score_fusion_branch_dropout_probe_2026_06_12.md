# SCP-Fusion Branch-Dropout Probe

Run date: 2026-06-12

This follow-up tests one proposed SCP-Fusion v1 idea from the research roadmap: branch dropout in the score-fusion head. The goal is to prevent the saved-score logistic stack from over-relying on one branch when crossing from Ishu AI-vs-real to source-balanced MS COCOAI.

Implementation:

- `scripts/fuse_prediction_scores.py`
- new options:
  - `--branch-dropout-rate`
  - `--branch-dropout-repeats`
  - `--branch-dropout-fill`
  - `--fusion-c`
- new artifact:
  - `score_fusion_coefficients.csv`, with standardized and raw-score coefficients.

Probe configuration:

- source dataset: Ishu AI-vs-real 2026;
- target dataset: source-balanced MS COCOAI / Defactify validation;
- seeds: 7, 17, 29;
- base branches: `combined_v3`, ResNet-18, physics-guided ResNet-18 + `combined_v3`, frozen ConvNeXt-Tiny;
- branch dropout: rate `0.25`, repeats `8`, neutral fill `0.5`.

## Main Result

Branch dropout is a useful negative result. It very slightly raises default-threshold accuracy, but it hurts ranking and probability quality.

| method | mean accuracy | mean AUC | mean Brier | mean ECE | predicted fake rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| SCP-Fusion v0 | 0.5910 | 0.7282 | 0.3190 | 0.3087 | 0.1383 |
| branch-dropout fusion | 0.5923 | 0.7250 | 0.3389 | 0.3325 | 0.1403 |

The accuracy change is only `+0.0013`, while AUC drops by `0.0032` and Brier/ECE both worsen. That means score-level branch dropout should not be claimed as SCP-Fusion v1 yet.

## Coefficients

The coefficient export is still useful for explainability. Averaged across the three branch-dropout seeds, the standardized branch weights are:

| branch | mean standardized coefficient | mean raw-score coefficient |
| --- | ---: | ---: |
| frozen ConvNeXt-Tiny | 1.5361 | 3.7817 |
| physics-guided fusion | 0.8392 | 2.1665 |
| `combined_v3` | 0.7137 | 2.1843 |
| ResNet-18 | 0.6049 | 1.5660 |

All branches remain positive, but frozen ConvNeXt-Tiny is still the dominant ranking branch. The physics-guided and handcrafted branches remain useful complementary evidence, especially for calibration and source-threshold behavior.

## Interpretation

This narrows the next SCP-Fusion step:

- Branch dropout alone does not fix the conservative target-domain fake scores.
- The next improvement should focus on source-aware calibration or validation, not only score-stack regularization.
- The coefficient export should stay: it gives WIFS/DFF reviewers a compact explanation of which branch the fusion head used.
- `combined_v4` should be added to the score-fusion branch list only after the larger repeated-seed and source-heldout runs confirm it is stable.

## Artifacts

Checked-in compact assets:

- `reports/assets/score_fusion_branch_dropout_calibration_summary.csv`
- `reports/assets/score_fusion_branch_dropout_calibration_metrics.csv`
- `reports/assets/score_fusion_branch_dropout_coefficients.csv`
- `reports/assets/score_fusion_branch_dropout_coefficient_summary.csv`

Local run folders are ignored by Git and can be regenerated under `runs/score_fusion_branch_dropout/`.

## Reproduce

Seed 7 example:

```powershell
python scripts/fuse_prediction_scores.py `
  --out-dir runs\score_fusion_branch_dropout\ishu_seed7_to_ms_cocoai_all4 `
  --seed 7 `
  --branch-dropout-rate 0.25 `
  --branch-dropout-repeats 8 `
  --branch-dropout-fill neutral `
  --train combined_v3=runs\ishu_ai_vs_real_2026_initial\feature_combined_v3_logistic_regression\predictions.csv `
  --train resnet18=runs\ishu_ai_vs_real_2026_initial\resnet18\predictions.csv `
  --train physics_guided=runs\ishu_ai_vs_real_2026_physics_guided_seed7\physics_guided_resnet18_combined_v3\predictions.csv `
  --train convnext_tiny_frozen=runs\ishu_ai_vs_real_2026_frozen_encoder\convnext_tiny_seed7\predictions.csv `
  --variant ms_cocoai:combined_v3=runs\ishu_to_ms_cocoai_source_balanced_seed7\combined_v3\predictions.csv `
  --variant ms_cocoai:resnet18=runs\ishu_to_ms_cocoai_source_balanced_seed7\resnet18\predictions.csv `
  --variant ms_cocoai:physics_guided=runs\ishu_to_ms_cocoai_source_balanced_seed7\physics_guided_resnet18_combined_v3\predictions.csv `
  --variant ms_cocoai:convnext_tiny_frozen=runs\ishu_to_ms_cocoai_source_balanced_seed7\convnext_tiny_frozen\predictions.csv
```

Then compare branch-dropout predictions against SCP-Fusion v0:

```powershell
python scripts/summarize_calibration_metrics.py `
  --out-dir runs\calibration_diagnostics\score_fusion_branch_dropout_probe `
  --n-bins 10 `
  --predictions seed7:scp_fusion_v0=runs\score_fusion\ishu_seed7_to_ms_cocoai_all4\ms_cocoai\predictions.csv `
  --predictions seed17:scp_fusion_v0=runs\score_fusion\ishu_seed17_to_ms_cocoai_all4\ms_cocoai\predictions.csv `
  --predictions seed29:scp_fusion_v0=runs\score_fusion\ishu_seed29_to_ms_cocoai_all4\ms_cocoai\predictions.csv `
  --predictions seed7:scp_fusion_branch_dropout=runs\score_fusion_branch_dropout\ishu_seed7_to_ms_cocoai_all4\ms_cocoai\predictions.csv `
  --predictions seed17:scp_fusion_branch_dropout=runs\score_fusion_branch_dropout\ishu_seed17_to_ms_cocoai_all4\ms_cocoai\predictions.csv `
  --predictions seed29:scp_fusion_branch_dropout=runs\score_fusion_branch_dropout\ishu_seed29_to_ms_cocoai_all4\ms_cocoai\predictions.csv
```
