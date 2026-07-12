# FREUID Code Freeze Status - 2026-07-13

This note records the FREUID Challenge state at the start of code-freeze day in the local timezone.

## Frozen Candidate

- Best Kaggle submission: `54511333`
- Public score: `0.37009`
- Submission message: `12k four-way fusion v3: combined_v4 plus CUDA ConvNeXt with metadata fallback`
- Submitted candidate CSV: `outputs/freuid_2026/public_12k_fourway_fusion_submission_packaged/submission.csv`
- Candidate rows: `142,818`
- Lint state: pass, exact `id,label` sample order, finite score labels in `[0, 1]`

## Frozen Runtime

The frozen scoring formula is:

```text
fraud_score = 0.7 * combined_v4_hgb(image) + 0.3 * convnext_tiny_logreg(image)
```

Frozen runtime release:

```text
https://github.com/DrStrangel0ve/ai-image-forensics-comparison/releases/tag/freuid-freeze-2026-07-10
```

Release assets:

- `freuid_frozen_stack_2026_07_10.zip`
- `freuid_short_report_draft_2026_07_10.pdf`
- `freuid_final_package_draft_2026_07_10.zip`

The runtime artifact release was created before code freeze and contains the frozen classifiers, ConvNeXt checkpoint cache, fusion summary, artifact manifest, and README.

## Validation Snapshot

Aligned `12,000` train / `4,000` validation split:

| Method | Accuracy | AUC | APCER @ 1% BPCER | AuDET proxy |
| --- | ---: | ---: | ---: | ---: |
| `combined_v4_hgb` | 0.8850 | 0.9584 | 0.2385 | 0.0417 |
| `convnext_tiny_logreg` | 0.8628 | 0.9457 | 0.2985 | 0.0550 |
| frozen score fusion | 0.8938 | 0.9661 | 0.2135 | 0.0341 |

Public leaderboard history:

| Submission | Public score | Note |
| --- | ---: | --- |
| `54503265` | 0.38042 | first image fusion with metadata fallback |
| `54505114` | 0.39303 | larger `combined_v4` image score candidate |
| `54511333` | 0.37009 | current best frozen score fusion |

## Package Verification

Current verifier:

```powershell
.\.venv\Scripts\python.exe scripts\verify_freuid_final_package.py
```

Latest heartbeat result: `pass`.

The verifier checks:

- final package zip integrity and required entries;
- submitted CSV lint against the official sample submission;
- report PDF header;
- pinned discussion draft content;
- required GitHub release assets.

## Docker Status

Local no-network Docker execution is still blocked by machine configuration, not repo code.

Observed blocker:

```text
HCS_E_HYPERV_NOT_INSTALLED
```

Meaning: WSL2 cannot start the `docker-desktop` distro because virtualization / Windows Virtual Machine Platform is unavailable.

Resolution note:

```text
reports/freuid_docker_blocker_resolution_2026_07_10.md
```

Post-fix smoke command:

```powershell
.\.venv\Scripts\python.exe scripts\smoke_test_freuid_docker.py
```

## Freeze Rules Going Forward

Allowed after this point:

- documentation edits;
- package assembly and upload logistics;
- Docker build/orchestration fixes that do not change model weights, model architecture, feature extraction, thresholds, or fusion weights;
- private-test inference using the frozen runtime.

Not allowed after this point:

- training or fine-tuning;
- changing classifiers, checkpoints, feature sets, preprocessing, hyperparameters, thresholds, or fusion weights;
- using private-test images for model updates;
- replacing the frozen candidate unless the rules explicitly allow it and the change was finalized before freeze.

## Current Next Action

Wait for the external Docker/WSL2 virtualization blocker to be fixed, then run the no-network Docker smoke test. Until that changes, the non-Docker package state is verified and ready.
