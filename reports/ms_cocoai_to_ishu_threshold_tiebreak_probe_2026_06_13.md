# MS COCOAI to Ishu Threshold Tie-Break Probe

Run date: 2026-06-13

This probe follows the held-out source-threshold fusion result. The previous `C=0.03` all-branch fusion selected thresholds on a held-out source slice using source accuracy, but it still over-called generated images on Ishu. The new `--threshold-tiebreak` option keeps the same source objective and changes only how equally good source thresholds are chosen.

## Setup

- source dataset: `data/raw/ms_cocoai_2026_validation_source_balanced_100/validation`
- target dataset: `data/raw/ishu_ai_vs_real_2026`
- seeds: `7`, `17`, `29`
- fusion branches: `combined_v3`, ResNet-18, physics-guided ResNet-18, ConvNeXt-Tiny, CLIP ViT-B/32, DINOv2-small
- fusion setting: `C=0.03`
- fit rows: 80% of aligned source rows
- threshold rows: held-out 20% of aligned source rows
- threshold objective: `source_accuracy`
- tested tie-breakers: `near_half`, `higher`, `lower`

## Reverse Target Results

Mean over three seeds on MS COCOAI -> Ishu:

| threshold tie-break | accuracy | AUC | Brier | ECE | precision | recall | predicted fake rate | selected threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `higher` | 0.7193 | 0.8291 | 0.2188 | 0.2060 | 0.6624 | 0.8869 | 0.6608 | 0.6280 |
| `lower` | 0.6988 | 0.8291 | 0.2188 | 0.2060 | 0.6394 | 0.9167 | 0.7105 | 0.5835 |
| `near_half` | 0.6959 | 0.8291 | 0.2188 | 0.2060 | 0.6378 | 0.9107 | 0.7076 | 0.5880 |

The conservative `higher` tie-breaker is the best current reverse operating point. It improves over the previous source-threshold fusion accuracy (`0.7193` vs `0.6959`) and reduces the target fake-call rate (`0.6608` vs `0.7076`) without changing the fused ranking or probability scores.

For paper framing, this is useful because it is still source-only: target labels are not used to choose the threshold. It does not fully solve calibration because the predicted fake rate remains above the true balanced target prior, but it shows that forensic operating preferences can be encoded in the source-threshold procedure.

Follow-up: `reports/ms_cocoai_to_ishu_threshold_cap_probe_2026_06_13.md` adds a source-side predicted-fake-rate cap. The best `0.48` cap raises the reverse operating-point accuracy to `0.7222` and lowers the target fake-call rate to `0.6228`.

## Interpretation

- Conservative tie-breaking is a simple utility-aware step that helps exactly where reverse fusion was failing.
- The result now beats the physics-guided ResNet-18 default accuracy by about 3.2 points (`0.7193` vs `0.6871`) while keeping much higher AUC (`0.8291` vs `0.7420`).
- Because AUC/Brier/ECE are unchanged, this should be described as operating-point selection, not a better score model.
- The next method step is still to train the fusion head with utility-aware validation or a fake-call-rate penalty, rather than only changing thresholds after training.

## Artifacts

- per-seed metrics: `reports/assets/ms_cocoai_to_ishu_threshold_tiebreak_metrics.csv`
- mean metrics: `reports/assets/ms_cocoai_to_ishu_threshold_tiebreak_mean_metrics.csv`
- local run directory: `runs/ms_cocoai_to_ishu_neural_fusion/`
