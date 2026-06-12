# SCP-Fusion Utility-Tuned Triage Probe

Run date: 2026-06-13

This follow-up tests the next step suggested by the source-heldout calibration stress test: tune calibration and triage jointly instead of reporting only fixed symmetric 5% and 10% operating points.

The new script, `scripts/summarize_source_holdout_triage_tuning.py`, searches score transforms and asymmetric two-threshold triage budgets on the calibration sources, then evaluates the selected operating point on a held-out generated source.

## Setup

Compared score-fusion variants:

- SCP-Fusion v0;
- branch-dropout fusion;
- source-calibrated fusion.

Search space:

- score modes: `raw`, `temperature_balanced`;
- max real-image FPR grid: `0.02, 0.05, 0.08, 0.10, 0.15`;
- max fake-clearance grid: `0.02, 0.05, 0.08, 0.10, 0.15`.

Selection objective:

```text
0.5 * fake_detection
+ 0.5 * real_clearance
- 1.0 * real_fpr
- 1.0 * fake_false_clearance
```

This gives undecided samples zero utility, rewards correct high-confidence fake and real calls, and penalizes the two forensic mistakes: calling real images fake and clearing generated images as likely real.

## Results

Means are over 15 source-heldout rows: 3 seeds times 5 generated MS COCOAI sources. Confidence intervals are deterministic 95% bootstrap intervals over those rows.

| method | selected modes | utility | coverage | triage accuracy | real FPR | fake false clearance | fake detection | real clearance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SCP-Fusion v0 | raw:15 | 0.1066 | 0.5029 | 0.7385 | 0.1232 | 0.1500 | 0.3980 | 0.3616 |
| branch-dropout fusion | raw:15 | 0.0891 | 0.4832 | 0.7306 | 0.1200 | 0.1527 | 0.3800 | 0.3435 |
| source-calibrated fusion | raw:15 | 0.0867 | 0.4867 | 0.7302 | 0.1216 | 0.1533 | 0.3747 | 0.3485 |

## Interpretation

The tuned result strengthens the earlier mixed finding:

- SCP-Fusion v0 is still the best triage geometry under source-heldout selection.
- Temperature-balanced calibration is helpful for Brier/ECE diagnostics, but it was not selected for any of the 15 utility-tuned triage folds.
- Source-calibrated fusion improves probability quality in the calibration report, yet it does not improve high-confidence source-heldout triage.
- The selected operating points are naturally asymmetric: mean selected max real FPR is about 0.09-0.095, while mean selected max fake clearance is about 0.13-0.14. A fixed symmetric 5%/10% table is useful for reporting, but the actual forensic operating point wants separate controls for real-image harm and fake-image false clearance.

For WIFS/DFF framing, this is a better claim than "calibration fixes transfer." The more defensible claim is: source-heldout evaluation exposes a split between probability calibration and triage utility, and SCP-Fusion should optimize/report both.

## Artifacts

Checked-in compact assets:

- `reports/assets/score_fusion_source_holdout_triage_tuned_summary.csv`
- `reports/assets/score_fusion_source_holdout_triage_tuned.csv`

The full candidate grid is intentionally left under ignored local runs because it is much larger than the selected tables:

- `runs/source_holdout_triage_tuning/score_fusion_compare/source_holdout_triage_grid.csv`

## Reproduce

```powershell
python scripts\summarize_source_holdout_triage_tuning.py `
  --metadata data\raw\ms_cocoai_2026_validation_source_balanced_100\metadata.csv `
  --out-dir runs\source_holdout_triage_tuning\score_fusion_compare `
  --split validation `
  --seed 7 `
  --max-real-fpr-grid 0.02,0.05,0.08,0.10,0.15 `
  --max-fake-clearance-grid 0.02,0.05,0.08,0.10,0.15 `
  --score-mode raw `
  --score-mode temperature_balanced `
  --predictions seed7:scp_fusion_v0=runs\score_fusion\ishu_seed7_to_ms_cocoai_all4\ms_cocoai\predictions.csv `
  --predictions seed17:scp_fusion_v0=runs\score_fusion\ishu_seed17_to_ms_cocoai_all4\ms_cocoai\predictions.csv `
  --predictions seed29:scp_fusion_v0=runs\score_fusion\ishu_seed29_to_ms_cocoai_all4\ms_cocoai\predictions.csv `
  --predictions seed7:branch_dropout=runs\score_fusion_branch_dropout\ishu_seed7_to_ms_cocoai_all4\ms_cocoai\predictions.csv `
  --predictions seed17:branch_dropout=runs\score_fusion_branch_dropout\ishu_seed17_to_ms_cocoai_all4\ms_cocoai\predictions.csv `
  --predictions seed29:branch_dropout=runs\score_fusion_branch_dropout\ishu_seed29_to_ms_cocoai_all4\ms_cocoai\predictions.csv `
  --predictions seed7:source_calibrated=runs\score_fusion_source_calibrated\ishu_seed7_to_ms_cocoai_all4\ms_cocoai\predictions.csv `
  --predictions seed17:source_calibrated=runs\score_fusion_source_calibrated\ishu_seed17_to_ms_cocoai_all4\ms_cocoai\predictions.csv `
  --predictions seed29:source_calibrated=runs\score_fusion_source_calibrated\ishu_seed29_to_ms_cocoai_all4\ms_cocoai\predictions.csv
```
