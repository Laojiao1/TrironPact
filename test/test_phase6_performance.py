"""性能协议关键顺序与统计边界。"""

import pytest

from bench.performance import interleaved_order, stats


def test_interleaving_rotates_and_reverses_order():
    orders = interleaved_order(("A", "B", "C"), 4)
    assert all(set(order) == {"A", "B", "C"} for order in orders)
    assert len(set(orders)) > 1
    assert orders[0] != orders[1]


def test_invalid_timing_rejected_before_speedup():
    with pytest.raises(ValueError):
        stats([1.0, float("nan")])
    assert stats([10.0, 20.0, 30.0])["median_us"] == 20.0
