import json

import pytest

from bench.e4.cross_hardware import aggregate
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


def test_cross_hardware_aggregate_requires_distinct_matching_complete_machines(tmp_path):
    paths = []
    for machine, gpu in (("host-a", "GPU A"), ("host-b", "GPU B")):
        row = {
            "hostname": machine,
            "source_fingerprint": "frozen",
            "go_machine": True,
            "inductor_performance": {
                "environment": {"gpu": gpu},
                "checks": {"raw_samples_complete": True},
            },
        }
        path = tmp_path / f"{machine}.json"
        path.write_text(json.dumps(row), encoding="utf-8")
        paths.append(path)
    report = aggregate(paths)
    assert report["go_e4_cross_hardware"] is True
    assert all(report["checks"].values())
