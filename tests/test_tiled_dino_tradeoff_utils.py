from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.tiled_dino_tradeoff_utils import (
    load_tiled_dino_tradeoff_summary,
    tiled_dino_evidence_summary,
    tiled_dino_sentence,
)


def test_tiled_dino_tradeoff_summary_formats_metric_claims(tmp_path: Path) -> None:
    tradeoff_path = tmp_path / "tradeoff.csv"
    _write_tradeoff(
        tradeoff_path,
        variants=["blur1", "jpeg30"],
        include_tile_max=True,
    )

    summary = load_tiled_dino_tradeoff_summary(tradeoff_path)

    assert summary == {
        "n_transforms": "2",
        "tile_max_acc_delta": "+0.0140",
        "tile_max_auc_delta": "+0.0160",
        "tile_mean_brier_count": "2/2",
        "tile_mean_ece_count": "1/2",
    }
    assert (
        tiled_dino_sentence(summary)
        == "Across 2 transform-stress probes, tiled-DINO `tile_max` gives +0.0140 accuracy and +0.0160 AUC average deltas, while `tile_mean` improves Brier on 2/2 and ECE on 1/2 transforms."
    )
    assert "tile_max: mean acc_delta=+0.0140" in tiled_dino_evidence_summary(summary)
    assert "tile_mean: Brier improves on 2/2" in tiled_dino_evidence_summary(summary)


def test_tiled_dino_tradeoff_summary_allows_missing_optional_path(tmp_path: Path) -> None:
    assert load_tiled_dino_tradeoff_summary(tmp_path / "missing.csv", allow_missing=True) is None


def test_tiled_dino_tradeoff_summary_rejects_missing_columns(tmp_path: Path) -> None:
    tradeoff_path = tmp_path / "tradeoff.csv"
    pd.DataFrame({"variant": ["blur1"], "score_mode": ["tile_max"]}).to_csv(tradeoff_path, index=False)

    with pytest.raises(ValueError, match="Missing tiled-DINO tradeoff columns"):
        load_tiled_dino_tradeoff_summary(tradeoff_path)


def test_tiled_dino_tradeoff_summary_rejects_missing_required_mode(tmp_path: Path) -> None:
    tradeoff_path = tmp_path / "tradeoff.csv"
    _write_tradeoff(
        tradeoff_path,
        variants=["blur1", "jpeg30"],
        include_tile_max=False,
    )

    with pytest.raises(ValueError, match="Missing tiled-DINO score_mode='tile_max'"):
        load_tiled_dino_tradeoff_summary(tradeoff_path)


def _write_tradeoff(tradeoff_path: Path, *, variants: list[str], include_tile_max: bool) -> None:
    mode_specs = [
        ("global", 0.0, 0.0, False, False),
        ("tile_mean", 0.002, 0.004, True, None),
    ]
    if include_tile_max:
        mode_specs.append(("tile_max", 0.014, 0.016, False, False))

    rows = []
    for index, variant in enumerate(variants):
        for score_mode, acc_delta, auc_delta, brier_improved, ece_improved in mode_specs:
            rows.append(
                {
                    "variant": variant,
                    "score_mode": score_mode,
                    "target_accuracy_mean_delta_vs_global": acc_delta,
                    "target_roc_auc_mean_delta_vs_global": auc_delta,
                    "target_brier_score_mean_improved_vs_global": brier_improved,
                    "target_expected_calibration_error_mean_improved_vs_global": (
                        index == 0 if ece_improved is None else ece_improved
                    ),
                }
            )
    pd.DataFrame(rows).to_csv(tradeoff_path, index=False)
