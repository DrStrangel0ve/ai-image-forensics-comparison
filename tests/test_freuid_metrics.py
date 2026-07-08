from __future__ import annotations

import pytest

from forensic_compare.freuid import apcer_at_bpcer, audet_proxy, freuid_metrics


def test_apcer_at_one_percent_bpcer_uses_strict_bona_fide_operating_point() -> None:
    y_true = [0, 0, 0, 1, 1, 1]
    scores = [0.10, 0.20, 0.80, 0.40, 0.60, 0.90]

    point = apcer_at_bpcer(y_true, scores, bpcer_target=0.01)

    assert point.bpcer == 0.0
    assert point.apcer == pytest.approx(2 / 3)
    assert point.n_bona_fide == 3
    assert point.n_attack == 3
    assert point.threshold > 0.80


def test_freuid_metrics_include_proxy_and_operating_point() -> None:
    y_true = [0, 0, 0, 1, 1, 1]
    scores = [0.05, 0.15, 0.25, 0.65, 0.75, 0.95]

    metrics = freuid_metrics(y_true, scores)

    assert metrics["apcer_at_1pct_bpcer"] == 0.0
    assert metrics["bpcer_at_operating_point"] == 0.0
    assert metrics["audet_proxy"] == audet_proxy(y_true, scores)
    assert metrics["n_bona_fide"] == 3
    assert metrics["n_attack"] == 3


def test_freuid_metrics_reject_missing_class() -> None:
    with pytest.raises(ValueError, match="both label classes"):
        apcer_at_bpcer([0, 0, 0], [0.1, 0.2, 0.3])
