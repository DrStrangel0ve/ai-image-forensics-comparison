# SCP-Fusion Source-Heldout Calibration Probe

Run date: 2026-06-13

This follow-up stress-tests the new source-calibrated SCP-Fusion path under the stricter MS COCOAI source-heldout diagnostics. The previous source-calibration probe showed better default-threshold accuracy and probability quality on the full source-balanced target split. This report asks whether that improvement survives when one generated source is held out at a time.

Compared methods:

- SCP-Fusion v0;
- branch-dropout fusion;
- source-calibrated fusion with source-domain class-balanced temperature scaling.

## Source-Heldout Post-Hoc Calibration

Each row holds out one generated MS COCOAI source, fits class-balanced temperature scaling on the other generated sources plus a real-image calibration split, and evaluates on the held-out generated source plus real test split. Means are over 15 held-out source/seed rows.

| method | calibrated accuracy | calibrated AUC | calibrated Brier | calibrated ECE | real FPR | fake detection |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| source-calibrated fusion | 0.7415 | 0.7082 | 0.1977 | 0.1268 | 0.0747 | 0.2820 |
| SCP-Fusion v0 | 0.7417 | 0.7123 | 0.1995 | 0.1331 | 0.0533 | 0.2293 |
| branch-dropout fusion | 0.7446 | 0.7092 | 0.2000 | 0.1340 | 0.0507 | 0.2327 |

Source-calibrated fusion has the best Brier score, ECE, and held-out fake detection, but it pays with a higher real-image false-positive rate and a small AUC drop. Branch dropout has the best calibrated accuracy here, but not the best probability quality.

## Triage Operating Points

At the strict 5% calibration error budget, source-calibrated fusion ties SCP-Fusion v0 on triage accuracy while using a smaller decided set.

| method | coverage | triage accuracy | real FPR | fake false clearance | fake detection |
| --- | ---: | ---: | ---: | ---: | ---: |
| source-calibrated fusion | 0.1975 | 0.7476 | 0.0453 | 0.0540 | 0.2227 |
| SCP-Fusion v0 | 0.2143 | 0.7476 | 0.0520 | 0.0527 | 0.2467 |
| branch-dropout fusion | 0.2126 | 0.7373 | 0.0547 | 0.0520 | 0.2400 |

At the looser 10% calibration error budget, SCP-Fusion v0 remains the better triage operating point.

| method | coverage | triage accuracy | real FPR | fake false clearance | fake detection |
| --- | ---: | ---: | ---: | ---: | ---: |
| SCP-Fusion v0 | 0.4373 | 0.7160 | 0.1307 | 0.1060 | 0.4093 |
| branch-dropout fusion | 0.4326 | 0.7107 | 0.1333 | 0.1027 | 0.3967 |
| source-calibrated fusion | 0.4309 | 0.7001 | 0.1387 | 0.1033 | 0.4047 |

## Interpretation

This is a useful mixed result:

- Source-domain calibration improves full-target default accuracy and probability quality.
- Under source-heldout post-hoc calibration, source-calibrated fusion improves Brier/ECE and fake detection, which supports the calibration direction.
- For two-threshold triage, source-calibrated fusion does not dominate SCP-Fusion v0. The v0 score geometry still gives better coverage at the same strict accuracy, and better triage accuracy at the 10% operating point.
- The next SCP-Fusion step should tune calibration and triage jointly instead of optimizing only Brier/ECE.

For the paper, this makes the calibration story more honest: calibration helps, but different forensic operating modes prefer different score geometries.

## Artifacts

Checked-in compact assets:

- `reports/assets/score_fusion_source_holdout_calibration_summary.csv`
- `reports/assets/score_fusion_source_holdout_calibration_detail.csv`
- `reports/assets/score_fusion_source_holdout_triage_5pct.csv`
- `reports/assets/score_fusion_source_holdout_triage_10pct.csv`

Local run folders are ignored by Git and can be regenerated under:

- `runs/source_holdout_calibration/score_fusion_source_calibrated_compare/`
- `runs/source_holdout_triage/score_fusion_source_calibrated_compare_5pct/`
- `runs/source_holdout_triage/score_fusion_source_calibrated_compare_10pct/`

## Reproduce

Calibration summary:

```powershell
python scripts/summarize_source_holdout_calibration.py `
  --metadata data\raw\ms_cocoai_2026_validation_source_balanced_100\metadata.csv `
  --out-dir runs\source_holdout_calibration\score_fusion_source_calibrated_compare `
  --split validation `
  --seed 7 `
  --n-bins 10 `
  --calibrator temperature_balanced `
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

For triage, use the same prediction arguments with `scripts/summarize_source_holdout_triage.py`, `--score-mode raw`, `--score-mode temperature_balanced`, and either `--max-real-fpr 0.05 --max-fake-clearance 0.05` or `--max-real-fpr 0.10 --max-fake-clearance 0.10`.
