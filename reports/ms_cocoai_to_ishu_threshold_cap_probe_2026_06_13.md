# MS COCOAI to Ishu Threshold Cap Probe

Run date: 2026-06-13

This probe extends the conservative threshold tie-break result with an explicit source-side cap on the selected threshold's predicted fake rate. The cap is still target-free: thresholds are selected only on the held-out MS COCOAI source slice, then applied unchanged to Ishu.

## Setup

- source dataset: `data/raw/ms_cocoai_2026_validation_source_balanced_100/validation`
- target dataset: `data/raw/ishu_ai_vs_real_2026`
- seeds: `7`, `17`, `29`
- fusion branches: `combined_v3`, ResNet-18, physics-guided ResNet-18, ConvNeXt-Tiny, CLIP ViT-B/32, DINOv2-small
- fusion setting: `C=0.03`
- fit rows: 80% of aligned source rows
- threshold rows: held-out 20% of aligned source rows
- threshold objective: `source_accuracy`
- tie-breaker: `higher`
- source predicted-fake-rate caps: `0.50`, `0.48`, `0.45`

## Reverse Target Results

Mean over three seeds on MS COCOAI -> Ishu:

| source fake-rate cap | accuracy | AUC | precision | recall | target fake-call rate | selected threshold | selected source fake rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.48` | 0.7222 | 0.8291 | 0.6711 | 0.8512 | 0.6228 | 0.6657 | 0.4750 |
| `0.45` | 0.7193 | 0.8291 | 0.6717 | 0.8393 | 0.6140 | 0.6847 | 0.4500 |
| `0.50` | 0.7193 | 0.8291 | 0.6624 | 0.8869 | 0.6608 | 0.6280 | 0.4917 |
| no cap, conservative tie-break | 0.7193 | 0.8291 | 0.6624 | 0.8869 | 0.6608 | 0.6280 | n/a |

The `0.48` cap is the best current reverse operating point. It slightly improves accuracy over conservative tie-breaking alone (`0.7222` vs `0.7193`) and cuts target fake-call rate further (`0.6228` vs `0.6608`) while preserving the same fused ranking. The stricter `0.45` cap reduces fake calls a bit more but starts trading away enough recall to lose the small accuracy gain.

## Interpretation

- A source-side predicted-fake-rate cap is a lightweight utility-aware threshold policy.
- The best capped result beats the physics-guided ResNet-18 default accuracy by about 3.5 points (`0.7222` vs `0.6871`) while keeping much higher AUC (`0.8291` vs `0.7420`).
- This is still operating-point selection, not a better scoring model: AUC/Brier/ECE remain unchanged.
- The next model-side step is to train SCP-Fusion with a validation objective that internalizes this fake-call-rate preference rather than applying it after fitting.

## Artifacts

- per-seed metrics: `reports/assets/ms_cocoai_to_ishu_threshold_cap_metrics.csv`
- mean metrics: `reports/assets/ms_cocoai_to_ishu_threshold_cap_mean_metrics.csv`
- local run directory: `runs/ms_cocoai_to_ishu_neural_fusion/`
