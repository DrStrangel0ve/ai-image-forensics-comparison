import pytest

from scripts.run_repeated_benchmark import _clean_forwarded_args


def test_clean_forwarded_args_strips_separator() -> None:
    assert _clean_forwarded_args(["--", "--methods", "combined_v3", "neural"]) == [
        "--methods",
        "combined_v3",
        "neural",
    ]


def test_clean_forwarded_args_rejects_reserved_values() -> None:
    with pytest.raises(SystemExit, match="--seed"):
        _clean_forwarded_args(["--", "--methods", "neural", "--seed", "17"])

    with pytest.raises(SystemExit, match="--out-dir"):
        _clean_forwarded_args(["--methods", "neural", "--out-dir=runs/example"])
