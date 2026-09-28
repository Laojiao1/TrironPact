"""统计分母、安全配对与变异预算的防回归测试。"""

import math

import pytest

from bench.metrics import decision_metrics, paired_round_interval, paired_speedups, select_witnesses


def test_unknown_does_not_become_true_negative_or_false_negative():
    rows = [
        {"id": "a", "kernel": "A", "truth": "eligible", "truth_source": "audit:a", "decision": "True"},
        {"id": "b", "kernel": "A", "truth": "ineligible", "truth_source": "audit:b", "decision": "False"},
        {"id": "c", "kernel": "B", "truth": "eligible", "truth_source": "audit:c", "decision": "Unknown"},
        {"id": "d", "kernel": "B", "truth": "unlabeled", "decision": "False"},
    ]
    result = decision_metrics(rows)
    assert result["counts"] == {"total": 4, "unlabeled": 1, "labeled": 3, "decided": 2,
                                "tp": 1, "fp": 0, "tn": 1, "fn": 0, "Unknown": 1, "Unsupported": 0}
    assert result["decision_coverage"] == pytest.approx(2 / 3)
    assert result["recall_on_decided"] == 1
    assert result["by_kernel"]["B"]["unknown"] == 1
    assert result["macro_by_kernel"]["recall_on_decided"]["defined_kernels"] == 1
    with pytest.raises(ValueError, match="审计来源"):
        decision_metrics([{"id": "x", "truth": "eligible", "decision": "True"}])


def test_speedup_excludes_unsafe_or_unverified_paths():
    rows = [
        {"id": "safe", "same_semantics": True, "paths": {"fallback": {"safe": True, "median_us": 20}, "dispatch": {"safe": True, "median_us": 10}}},
        {"id": "unsafe", "same_semantics": True, "paths": {"fallback": {"safe": True, "median_us": 10}, "dispatch": {"safe": False, "median_us": 1}}},
        {"id": "bad_time", "same_semantics": True, "paths": {"fallback": {"safe": True, "median_us": 10}, "dispatch": {"safe": True, "median_us": math.nan}}},
    ]
    result = paired_speedups(rows, "fallback", "dispatch")
    assert result["paired_count"] == 1
    assert result["geomean"] == 2
    assert len(result["excluded"]) == 2


def test_witness_budget_prefers_flips_and_is_deterministic():
    rows = [{"id": "later", "kernel": "A", "reason": "sample"},
            {"id": "flip", "kernel": "A", "reason": "decision_flip"},
            {"id": "boundary", "kernel": "B", "reason": "tile_boundary"},
            {"id": "flip", "kernel": "A", "reason": "decision_flip"}]
    result = select_witnesses(rows, max_per_kernel=1, max_total=2)
    assert [row["id"] for row in result["selected"]] == ["flip", "boundary"]
    assert result["unique"] == 3


def test_paired_round_interval_is_conditional_and_reproducible():
    rows = [{"same_semantics": True, "paths": {"a": {"safe": True, "samples_us": [20.0] * 8},
                                                "b": {"safe": True, "samples_us": [10.0] * 8}}}]
    result = paired_round_interval(rows, "a", "b", draws=100)
    assert result["point_geomean"] == pytest.approx(2)
    assert result["ci95"] == pytest.approx((2, 2))
