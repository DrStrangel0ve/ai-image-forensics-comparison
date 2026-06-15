# Manuscript Assembly Map

Run date: 2026-06-14

Generated bridge from the checked-in section drafts, paper skeletons, figures, and LaTeX tables to concrete WIFS/DFF writing tasks.

## Venue Summary

| venue | sections | target_pages | source_word_count | assets_present | assets_total |
| --- | --- | --- | --- | --- | --- |
| DFF | 6 | 6.6 | 1401 | 6 | 6 |
| WIFS | 7 | 5.75 | 1167 | 7 | 7 |

## WIFS Assembly

| order | section | target_pages | source_sections | source_word_count | primary_assets | writing_action |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Abstract and contribution box | 0.25 | WIFS compact abstract | 189 | reports/submission_text_drafts_2026_06_14.md; reports/wifs_breadth_decision_2026_06_14.md | Use the compact abstract; keep the contribution wording as source-heldout benchmark evidence, not SOTA. |
| 2 | Introduction | 0.75 | WIFS Introduction Draft | 175 | reports/paper_section_drafts_2026_06_14.md; reports/assets/latex_tables/same_domain_anchor.tex; reports/assets/latex_tables/transfer_frontier.tex | Open with ranking/calibration/threshold shift; cite physics-guided same-domain gain and CLIP transfer frontier. |
| 3 | Related work and problem framing | 0.75 | WIFS Related Work Draft | 99 | references.bib; reports/literature_map_2026_06_14.md | Compress to forensic benchmark families; preserve the single-image physical-proxy caveat. |
| 4 | Data, audits, and metrics | 0.75 | WIFS Data And Audit Draft | 139 | reports/assets/latex_tables/source_holdout_stress.tex; reports/source_holdout_generator_stress_2026_06_14.md | Explain Ishu, source-balanced MS COCOAI, source labels, calibration, fake-call rate, and triage coverage. |
| 5 | Methods | 1.0 | WIFS Methods Draft | 101 | reports/method_family_comparison_2026_06_14.md; reports/assets/latex_tables/method_family_comparison.tex | Describe method families rather than every script; keep SCP-Fusion as a diagnostic protocol. |
| 6 | Results | 1.75 | WIFS Results Draft | 336 | reports/assets/latex_tables/transfer_frontier.tex; reports/assets/latex_tables/reverse_operating_points.tex; reports/assets/latex_tables/calibration_operating_modes.tex; reports/assets/latex_tables/robustness_stress.tex; reports/assets/latex_tables/reconstruction_ablation.tex; reports/assets/publication_score_fusion_clip_frontier.png; reports/assets/publication_reverse_operating_points.png | Use compact tables first; include only two or three figures if the official page limit gets tight. |
| 7 | Limitations and reproducibility | 0.5 | Limitations And Reproducibility Draft | 128 | reports/reproducibility_checklist_2026_06_12.md; reports/submission_package_lint_2026_06_14.md | State that raw datasets/models are external while reports, commands, tables, figures, and lints are checked in. |

## DFF Assembly

| order | section | target_pages | source_sections | source_word_count | primary_assets | writing_action |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Abstract and diagnostic claim | 0.35 | DFF workshop abstract | 219 | reports/submission_text_drafts_2026_06_14.md; reports/method_family_comparison_2026_06_14.md | Frame SCP-Fusion as a diagnostic protocol for source shift, not a universal detector. |
| 2 | Motivation and related work | 1.0 | WIFS Introduction Draft; WIFS Related Work Draft | 274 | references.bib; reports/research_deep_dive_2026_06_12.md | Expand the workshop motivation around deepfake forensics, generator shift, and processing robustness. |
| 3 | Protocol and branches | 1.25 | WIFS Data And Audit Draft; WIFS Methods Draft | 240 | reports/assets/latex_tables/method_family_comparison.tex; reports/assets/claim_evidence_matrix.md | Keep branch descriptions inspectable: physical/signal, neural, foundation, reconstruction, and score/source fusion. |
| 4 | Main results and operating points | 1.75 | WIFS Results Draft | 336 | reports/assets/latex_tables/transfer_frontier.tex; reports/assets/latex_tables/reverse_operating_points.tex; reports/assets/latex_tables/calibration_operating_modes.tex; reports/assets/publication_score_fusion_clip_frontier.png; reports/assets/publication_triage_operating_points.png | Carry the multi-objective story: CLIP ranks best, physics helps same-domain, source-aware fusion helps reverse operating points. |
| 5 | Failure analysis and ablations | 1.5 | DFF Expansion Draft | 204 | reports/assets/qualitative_seed29_scp_fusion_false_negatives.png; reports/combined_v4_source_slice_diagnostics_2026_06_13.md; reports/assets/latex_tables/source_holdout_stress.tex; reports/assets/latex_tables/reconstruction_ablation.tex | Use the selected qualitative grid and source-slice diagnostics to explain what the fused detector still misses. |
| 6 | Limitations, ethics, and reproducibility | 0.75 | Limitations And Reproducibility Draft | 128 | reports/reproducibility_checklist_2026_06_12.md; reports/submission_package_lint_2026_06_14.md | Keep the same overclaim guardrails and call out external datasets/model checkpoints. |

## Guardrails

- WIFS should stay compact and multi-objective: ranking, calibration, fake-call rate, source-aware decisions, robustness, and reproducibility.
- DFF can spend more room on failure analysis, source slices, and SCP-Fusion as a diagnostic protocol.
- Do not add new model claims unless they already appear in the checked-in claim-evidence matrix and lints.
- Keep combined_v4 and reconstruction_v2 as ablation evidence unless a later report explicitly promotes them.
