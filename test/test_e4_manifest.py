from bench.e4.manifest import build
from integration.specs import WORKLOADS, distinct_kernel_ids, workload_spec


def test_e4_workloads_have_distinct_traceable_kernel_bodies():
    report = build()
    assert report["go_manifest"]
    assert len(WORKLOADS) == 2
    assert len(distinct_kernel_ids()) >= 10
    assert report["counts"]["distinct_kernels"] == 14
    assert all(report["checks"].values())


def test_unknown_workload_fails_closed():
    try:
        workload_spec("missing")
    except ValueError as error:
        assert "Unknown" in str(error)
    else:
        raise AssertionError("未知工作负载不能隐式进入 e4a")
