from collections import Counter

from scripts.export_hf_image_dataset import _should_skip, _split_done


def test_source_cap_keeps_only_requested_fake_sources() -> None:
    counts = Counter({"real": 500, "ai_generated": 199})
    source_counts = Counter({"1": 100, "2": 99})

    assert _should_skip("ai_generated", "1", counts, source_counts, 500, None, 100, {"1", "2"})
    assert not _should_skip("ai_generated", "2", counts, source_counts, 500, None, 100, {"1", "2"})
    assert _should_skip("ai_generated", "3", counts, source_counts, 500, None, 100, {"1", "2"})


def test_source_balanced_split_done_requires_all_requested_sources() -> None:
    counts = Counter({"real": 500, "ai_generated": 500})
    source_counts = Counter({"1": 100, "2": 100, "3": 100, "4": 100, "5": 99})

    assert not _split_done(counts, source_counts, 500, None, 100, {"1", "2", "3", "4", "5"})

    source_counts["5"] = 100
    assert _split_done(counts, source_counts, 500, None, 100, {"1", "2", "3", "4", "5"})


def test_legacy_binary_class_caps_still_finish_each_class() -> None:
    counts = Counter({"real": 10, "ai_generated": 9})
    source_counts = Counter()

    assert not _split_done(counts, source_counts, 10, 10, None, set())

    counts["ai_generated"] = 10
    assert _split_done(counts, source_counts, 10, 10, None, set())
