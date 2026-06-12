from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

from scripts.run_feature_ablation_grid import FeatureConfig, _command, _configs, _parse_config


def test_parse_feature_ablation_config() -> None:
    config = _parse_config("v4k60=combined_v4:logistic_regression:60")

    assert config == FeatureConfig(
        name="v4k60",
        feature_set="combined_v4",
        classifier="logistic_regression",
        select_k=60,
    )


def test_parse_feature_ablation_config_rejects_bad_values() -> None:
    with pytest.raises(ValueError, match="NAME=FEATURE_SET"):
        _parse_config("combined_v4:logistic_regression:60")
    with pytest.raises(ValueError, match="integer"):
        _parse_config("bad=combined_v4:logistic_regression:nope")
    with pytest.raises(ValueError, match="non-negative"):
        _parse_config("bad=combined_v4:logistic_regression:-1")


def test_feature_ablation_grid_default_configs() -> None:
    configs = _configs([])

    assert [config.name for config in configs] == [
        "combined_v3_logreg",
        "combined_v4_logreg",
        "combined_v4_logreg_selectk60",
        "combined_v4_logreg_selectk80",
    ]


def test_feature_ablation_grid_command_forwards_limits() -> None:
    args = Namespace(
        data_dir="data/raw/example",
        selection_score_func="f_classif",
        image_size=64,
        val_fraction=0.2,
        max_train_samples=120,
        max_test_samples=60,
        skip_errors=True,
    )
    command = _command(
        args,
        FeatureConfig("v4k60", "combined_v4", "logistic_regression", 60),
        seed=17,
        output_dir=Path("runs/grid/seed17/v4k60"),
    )

    assert "--feature-set" in command
    assert command[command.index("--feature-set") + 1] == "combined_v4"
    assert command[command.index("--select-k") + 1] == "60"
    assert command[command.index("--max-train-samples") + 1] == "120"
    assert "--skip-errors" in command
