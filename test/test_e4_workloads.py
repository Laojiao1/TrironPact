from integration.specs import distinct_kernel_ids


def test_workload_kernel_count_is_not_inflated_by_repeated_calls():
    ids = distinct_kernel_ids()
    assert len(ids) == len(set(ids)) == 14
    assert "liger_softmax" in ids
    assert "pytorch_double_strided" in ids


def test_workload_entrypoints_are_explicit():
    from integration.specs import WORKLOADS

    assert {item.kind for item in WORKLOADS} == {"compiler_graph_trace", "model_layer_trace"}
    assert all(item.entrypoint.startswith("integration.worker.") for item in WORKLOADS)
