"""新增离线基准审计的纯 CPU 汇总测试。"""

from bench.benchmark_audit import BENCH_IDS, summarize


def _meta(shape, stride):
    return {"shape": list(shape), "stride": list(stride), "storage_offset": 0}


def _workers(*, false_success: bool = False):
    workers = []
    for name in BENCH_IDS:
        if name.startswith("bench_feature_"):
            rows = [
                {"layout": "contiguous", "truth": "eligible", "observation": "correct", "mismatch_count": 0, "inputs": {"X": _meta((7, 11), (11, 1)), "F": _meta((11,), (1,))}},
                {"layout": "offset", "truth": "eligible", "observation": "correct", "mismatch_count": 0, "inputs": {"X": _meta((7, 11), (11, 1)), "F": _meta((11,), (1,))}},
                {"layout": "x_col_strided", "truth": "ineligible", "observation": "correct" if false_success else "numeric_mismatch", "mismatch_count": 0 if false_success else 10, "inputs": {"X": _meta((7, 11), (22, 2)), "F": _meta((11,), (1,))}},
                {"layout": "feature_strided", "truth": "ineligible", "observation": "correct" if false_success else "numeric_mismatch", "mismatch_count": 0 if false_success else 10, "inputs": {"X": _meta((7, 11), (11, 1)), "F": _meta((11,), (2,))}},
            ]
        else:
            inputs1 = {"X": _meta((129,), (1,))}
            inputs2 = {"X": _meta((129,), (2,))}
            if name in {"bench_multiply", "bench_axpy"}:
                inputs1["Z"] = _meta((129,), (1,))
                inputs2["Z"] = _meta((129,), (2,))
            rows = [
                {"layout": "contiguous", "truth": "eligible", "observation": "correct", "mismatch_count": 0, "inputs": inputs1},
                {"layout": "offset", "truth": "eligible", "observation": "correct", "mismatch_count": 0, "inputs": inputs1},
                {"layout": "strided", "truth": "ineligible", "observation": "correct" if false_success else "numeric_mismatch", "mismatch_count": 0 if false_success else 10, "inputs": inputs2},
            ]
        workers.append({"kernel": name, "exit_code": 0, "stderr": "", "detail": {"rows": rows}})
    return workers


def test_summarize_requires_expected_positive_and_negative_boundaries():
    report = summarize(_workers())
    assert report["go_offline_benchmark_audit"]


def test_summarize_rejects_false_success_on_strided_boundary():
    assert not summarize(_workers(false_success=True))["checks"]["ineligible_boundaries_witnessed"]
