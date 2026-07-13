# FREUID Code Freeze Status - 2026-07-13

## Frozen Candidates

Two complementary submissions are retained for final selection:

| Submission | Public score | Role |
| --- | ---: | --- |
| `54624136` | 0.25470 | strongest known-layout public specialist |
| `54627101` | 0.25799 | rank-fused unseen-type and capture candidate |

Both are explicitly selected in Kaggle (`2/2`) rather than left to automatic selection.

The OOD candidate changes all `7,821` locally available image predictions and changes zero of the `134,997` unavailable-image fallback values. Its small `0.00329` public-score cost is therefore a clean model comparison despite the organizer-reported public scoring issue.

## Frozen Runtime

The no-network runtime uses two checkpoints sequentially:

```text
template_score = ConvNeXt-Tiny(image at 224 px)
forensic_score = EfficientNet-B0(residual-adapted image at 384 px)
fraud_score = 0.85 * rank(template_score) + 0.15 * rank(forensic_score)
```

The forensic branch was trained with capture augmentation, balanced type/label sampling, and gradient-reversal document-type suppression. It achieved `24.0%` type accuracy on five document types while retaining `0.998826` AUC on the all-type validation split, showing that it no longer solves fraud detection by simply identifying a known template.

Runtime files:

- `scripts/infer_freuid_checkpoint_ensemble.py`
- `scripts/infer_freuid_finetune.py`
- `scripts/freeze_freuid_submission_artifacts.py`
- `docker/freuid/Dockerfile`
- `artifacts/freuid_2026/README.md`

The same frozen image supports `FREUID_VARIANT=public_specialist` for ref `54624136` and `FREUID_VARIANT=ood_rank` for ref `54627101`. This is inference orchestration only; checkpoints, model code, preprocessing, and weights are unchanged from model-freeze commit `a0802299603c05917008009a52fcad235aa4ea4d`.

The artifact manifest records checkpoint byte sizes and SHA-256 hashes. A local five-image run from the frozen artifact copies completed successfully on CUDA and wrote exact `id,label` output. Docker execution remains blocked only by the host's unavailable WSL2 virtualization (`HCS_E_HYPERV_NOT_INSTALLED`).

Organizer verification allows at most six hours on one A100 40GB with 24 CPUs. The exact frozen OOD path scored `1,000` images in `211.4 s` on an RTX 3080 Ti at local-safe batch size `32`; this includes substantial one-time Windows/model startup. The A100 container defaults to batch size `96` and `12` workers. Final compliance remains to be confirmed on organizer hardware.

## Domain-Shift Evidence

Random type-balanced validation is nearly saturated and is not the selection criterion. On the hard leave-EGYPT-document-type-out split:

| Evaluation | Method | APCER @ 1% BPCER | AuDET proxy | AUC |
| --- | --- | ---: | ---: | ---: |
| clean | global EfficientNet 384 | 0.404 | 0.171010 | 0.829248 |
| clean | frozen 85/15 rank ensemble | **0.402** | **0.167956** | **0.832336** |
| screenshot | global EfficientNet 384 | 0.422 | 0.171834 | 0.828460 |
| screenshot | frozen 85/15 rank ensemble | 0.416 | **0.168740** | **0.831586** |

The ensemble improves the ranking/AuDET objective under both clean and screenshot stress while preserving the strict low-BPCER operating point. Larger forensic weights improve average ranking further but increase APCER, so `0.15` is the conservative frozen weight.

## Freeze Rules

After the July 13 AoE code/model freeze, do not train, change checkpoints, alter preprocessing, or tune weights. Allowed work is documentation, packaging, Docker/orchestration repair, and inference with these frozen artifacts.

Organizer clarifications used for this decision:

- unseen private document types and additional capture conditions: <https://www.kaggle.com/competitions/the-freuid-challenge-2026-ijcai-ecai/discussion/704992>
- public leaderboard scoring update: <https://www.kaggle.com/competitions/the-freuid-challenge-2026-ijcai-ecai/discussion/723604>
