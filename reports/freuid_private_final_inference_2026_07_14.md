# FREUID Private Final Inference - 2026-07-14

## Freeze Boundary

This run performs inference only. It uses the unchanged frozen model commit
`fbe1fd910766c2702e052f6c46c9702580d92e69`, the same checkpoints,
preprocessing, rank function, and `0.85/0.15` OOD fusion. Commit
`f96996e` removes an invalid launcher argument; it does not change model
behavior.

## Verified Private Input

- Images: `134,997` JPEGs
- Bytes: `29,071,215,644`
- Source archive SHA-256: `97be8c26bfd4c8b73daa8aa9d4886b522d6d843e33dfeba977f3fdfc713b50fb`
- Sorted ID-list SHA-256: `3d90945183e39525e2e3596625fb560bbfc98edcf0d61c3d22c78e3ed9e8d32d`

## Frozen Inference

- Kaggle kernel: `arnavmalani/freuid-private-frozen-inference`, version 4
- Status: complete
- Compute: two Tesla T4 GPUs
- Wall time: `3,415.81 s`
- Public specialist checkpoint SHA-256: `33b7fe5abd6aa6740d0c9110fb3c923376160f38a823c87c0a61d227b7d800cb`
- Forensic checkpoint SHA-256: `f6c48d5594351569d2f0b5e7330ae07cb148c95095558c6bf6fb8c5de2188e0a`
- Private public-specialist scores SHA-256: `fecc786e7bdb5ce398cab95ff9480ac20e851d440d6483fd1dd88b666bd9650f`
- Private OOD-rank scores SHA-256: `394d90d04513409d82dcd2f90b631bac1876e4e01ea99f6421a16a222830c30a`

Both score files contain exactly `134,997` unique expected IDs, finite scores
in `[0, 1]`, and the verified private ID-list digest.

## Final Submissions

The merge replaced exactly the private rows and preserved the remaining
`7,821` frozen rows in their original order and text representation.

| Variant | Frozen base SHA-256 | Final SHA-256 | Kaggle ref | Public score |
| --- | --- | --- | ---: | ---: |
| public specialist | `35454097181d7430ea0e322e5c3dd8a73b2db3519a5215fc472b7c105000bae0` | `f2a8737665672f2638ef88b7cdab71c00168402b21076e9383cb5c22d6ca68b2` | `54673713` | 0.25470 |
| OOD rank 85/15 | `cbc3e6c0fbb0bbd9d35e7f4e1d33fc21835afa3f5f0b6f33b17824393fcf700d` | `5ce9667137ba83def3d9c139f4cd55e1d961a92c6bf42224020ec5b27b66df07` | `54673732` | 0.25799 |

Both final CSVs contain `142,818` rows and pass score-aware checks for exact
`id,label` columns, row count, ID set, ID order, uniqueness, and score range.
The unchanged public scores confirm that the preserved public rows match the
pre-private submissions.

## Final Selection

On 2026-07-14, the two pre-private selections were replaced in Kaggle. The
final-selection panel reports exactly `2/2`, with only refs `54673713` and
`54673732` checked. Refs `54624136` and `54627101` are no longer selected.

After confirming that no prior `arnavmalani` top-level reply existed, the
required reproducibility reply was posted exactly once on Kaggle thread
`718637` on 2026-07-14. The published reply records both final refs, timestamps,
CSV digests, frozen commit, public repository, and technical-report link.
