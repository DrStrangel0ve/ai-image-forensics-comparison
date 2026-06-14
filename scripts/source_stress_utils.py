from __future__ import annotations

from pathlib import Path

import pandas as pd


DEFAULT_SOURCE_STRESS_POLICY = "source_holdout_mean_utility_cap_0p48"


def load_source_stress_summary(
    path: Path | str,
    *,
    policy: str = DEFAULT_SOURCE_STRESS_POLICY,
    allow_missing: bool = False,
) -> dict[str, str] | None:
    path = Path(path)
    if not path.exists():
        if allow_missing:
            return None
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {
        "selection_policy",
        "heldout_source_name",
        "source_holdout_utility_mean",
        "source_holdout_recall_mean",
        "source_holdout_fake_miss_rate_mean",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing source-stress columns: {sorted(missing)}")
    policy_frame = frame[frame["selection_policy"].astype(str) == policy].copy()
    if policy_frame.empty:
        if allow_missing:
            return None
        raise ValueError(f"{path} has no rows for selection_policy={policy!r}")
    weakest = policy_frame.sort_values(
        ["source_holdout_utility_mean", "source_holdout_fake_miss_rate_mean"],
        ascending=[True, False],
    ).iloc[0]
    return {
        "policy": policy,
        "heldout_source_name": str(weakest["heldout_source_name"]),
        "utility": f"{float(weakest['source_holdout_utility_mean']):.4f}",
        "recall": f"{float(weakest['source_holdout_recall_mean']):.4f}",
        "fake_miss_rate": f"{float(weakest['source_holdout_fake_miss_rate_mean']):.4f}",
    }


def source_stress_sentence(summary: dict[str, str] | None) -> str:
    if summary is None:
        return "No held-out-generator stress summary was available for this build."
    return (
        f"For the paper-facing `{summary['policy']}` policy, the weakest held-out generator is "
        f"`{summary['heldout_source_name']}` with mean utility {summary['utility']}, "
        f"mean recall {summary['recall']}, and mean fake-miss rate {summary['fake_miss_rate']}."
    )
