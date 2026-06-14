# WIFS Breadth Decision

Run date: 2026-06-14

Decision: freeze the WIFS experimental scope and write the 6-page paper around the checked-in evidence. Do not make a new model run a WIFS blocker.

Current WIFS readiness: packet `ready_to_polish`, artifacts 50/50, lint reports 4/4, lint checks 192/192, claims ready 2, caveated 7, needs 0.

## Option Ranking

| option_id | decision | paper_use | manuscript_value | deadline_fit | reproducibility_fit | effort | risk | decision_score |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| freeze_current_scope | selected | WIFS main plan | 4.0 | 5.0 | 5.0 | 1.0 | 1.0 | 12.0 |
| source_aware_v4_selection | optional_spare_time | appendix ablation only | 4.0 | 4.0 | 3.0 | 2.0 | 3.0 | 6.0 |
| larger_source_split | defer | post-WIFS/DFF extension | 5.0 | 2.0 | 2.0 | 5.0 | 4.0 | 0.0 |
| true_tiled_neural_foundation | defer | post-WIFS/DFF extension | 5.0 | 2.0 | 2.0 | 5.0 | 4.0 | 0.0 |

## Evidence Anchors

| anchor | path | exists | use |
| --- | --- | --- | --- |
| method_family_comparison | reports/method_family_comparison_2026_06_14.md | True | Condense the physical, neural, foundation, reconstruction, and fusion families into the method/results spine. |
| source_holdout_stress | reports/source_holdout_generator_stress_2026_06_14.md | True | Use held-out generator stress as the breadth evidence for source-shift risk. |
| robustness_stress | reports/assets/latex_tables/robustness_stress.tex | True | Use transform stress as the processing-robustness table. |
| reconstruction_ablation | reports/assets/latex_tables/reconstruction_ablation.tex | True | Use reconstruction_v2 as a caveated ablation, not a new lead method. |
| combined_v4_source_slices | reports/combined_v4_source_slice_diagnostics_2026_06_13.md | True | Use v4 source slices as dataset-bias explanation for why v4 stays in the appendix. |

## Writing Rule

- Lead WIFS with metric-family breadth: same-domain anchor, transfer frontier, reverse operating points, robustness stress, source-heldout stress, and reconstruction ablation.
- Keep CLIP as the transfer ranking frontier and SCP-Fusion as a diagnostic protocol with calibrated operating points.
- Keep combined_v4 and reconstruction_v2 as caveated ablations; do not promote either to the main method.
- If spare compute appears, run source-aware v4 selection as an appendix-only add-on. Do not delay WIFS for a larger source split or true tiled neural/foundation branch.
