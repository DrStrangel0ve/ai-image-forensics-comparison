# FREUID Kaggle Submission Log - 2026-07-09

Generated during the first successful FREUID leaderboard upload session.

## Official Metric Check

The Kaggle Evaluation page states that `label` must be a numeric fraud score, not a hard class label:

- Higher score means more likely fraudulent.
- The FREUID Score combines AuDET and APCER at 1% BPCER.
- Lower public score is better.

This corrected the earlier local assumption that `label` had to be binary.

## Submissions

| Ref | Description | Public score | Notes |
| --- | --- | ---: | --- |
| 54481645 | metadata size baseline v1: released public sizes, conservative fallback | 1.00000 | Binary labels; valid upload but wrong output semantics for this metric. |
| 54481898 | metadata size score baseline v2: continuous fraud scores | 0.98832 | Correct score-valued format; weak metadata-only ranking. |
| 54481939 | metadata size score baseline v3: inverted direction check | 0.97578 | Inversion improved slightly, suggesting metadata direction/domain shift is weak but not enough. |

## Current Best

Best public score so far: `0.97578` from submission `54481939`.

This is not competitive yet. The public leaderboard top rows are near `0.0003`, so the next useful step is to acquire the official 17.58 GB image archive and run actual image-feature / neural scoring instead of metadata-only scoring.

## Data Access Notes

- Kaggle notebook CLI metadata still strips `competition_sources`, so the remote notebook launched with empty `/kaggle/input`.
- Browser Add Input found the competition after account access was granted, but the UI stalled at `Adding data source` and did not persist the input.
- The official Data page reports `77,189` files and `17.58 GB`.
- Kaggle API download attempts hit `429 Too Many Requests` after file-manifest pagination; retry after cooldown.

## Local Artifacts

- Score builder: `scripts/build_freuid_metadata_submission.py`
- Corrected linter: `scripts/lint_freuid_submission.py --allow-score-labels`
- Ignored generated outputs:
  - `outputs/freuid_2026/metadata_size_score_submission/`
  - `outputs/freuid_2026/metadata_size_score_inverted_submission/`
