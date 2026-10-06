import pytest

from bench.e4.diagnostics import paired_ratio_interval, percentile


def test_percentile_interpolates_and_rejects_empty_input():
    assert percentile([1.0, 2.0, 3.0], 0.5) == 2.0
    assert percentile([1.0, 3.0], 0.5) == 2.0
    with pytest.raises(ValueError, match="Unknown"):
        percentile([], 0.5)


def test_paired_interval_preserves_direction_and_pairing():
    lower = paired_ratio_interval([1.0, 1.1, 0.9, 1.0], [2.0, 2.2, 1.8, 2.0], draws=300)
    assert lower["stable_direction"] == "lower"
    assert lower["ci95"][1] < 1
    with pytest.raises(ValueError, match="Unknown"):
        paired_ratio_interval([1.0], [1.0, 2.0])
