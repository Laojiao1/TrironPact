import pytest

from pact.e3_mutation import BoundaryProbe, ordered_probes, probe_domain


def test_shared_domain_is_complete_and_stable():
    probes = probe_domain()
    assert len(probes) == 14
    assert len({item.id for item in probes}) == len(probes)
    assert len({item.fingerprint for item in probes}) == len(probes)
    assert {item.family for item in probes} == {
        "multi_axis_grid", "stride_broadcast_2d", "separate_inputs", "offset_alignment", "span_boundary", "reduction_row_length"
    }
    assert all(item.unreachable_reason for item in probes if not item.executable)


def test_methods_only_reorder_the_same_domain():
    wanted = {item.fingerprint for item in probe_domain()}
    for method in ("static_candidate", "uniform_random", "constrained_random", "predicate_guided"):
        rows = ordered_probes(method, 17)
        assert len(rows) == len(wanted)
        assert {item.fingerprint for item in rows} == wanted
    assert ordered_probes("uniform_random", 17) == ordered_probes("uniform_random", 17)


def test_invalid_or_unreachable_probe_is_rejected():
    with pytest.raises(ValueError):
        BoundaryProbe("bad", "k", "s", "span_boundary", "bad", ("x",), 1, executable=False)
    with pytest.raises(ValueError):
        BoundaryProbe("bad", "k", "s", "span_boundary", "bad", ("x",), 1, "case", False, "illegal")

