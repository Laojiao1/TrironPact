"""第五阶段：候选 Guard、真实输出、修复与独立分派。"""

from dataclasses import replace

import pytest
import torch

from pact.cost_model import CostEntry, CostTable, layout_family, shape_bucket
from pact.dispatch import dispatch, get_plan
from pact.guard_plan import KernelSpec, SPECS, compile_guard_plan
from pact.relayout import try_relayout
from scenarios.alignment_fixtures import index_hint, pointer_hint, run_index_hint, run_pointer_hint
from scenarios.cases import make_case, reference
from pact.semantics import SEMANTICS


@pytest.mark.parametrize("name", tuple(SPECS))
def test_all_plans_cover_accesses(name):
    plan = compile_guard_plan(name)
    assert plan.status == "Supported", plan.reason
    assert {check.stage for check in plan.checks} >= {"metadata", "span"}
    assert plan.fingerprint and len(plan.fingerprint) == 64
    assert all(check.candidate.origin and check.candidate.domain and check.candidate.binding for check in plan.checks)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("name,layout", [("A", "contiguous"), ("A2", "padded"), ("B", "offset"),
                                          ("D", "contiguous")])
def test_direct_fast(name, layout):
    v = make_case(name, layout, n=129 if name in ("B", "D") else 11)
    x, y = v if isinstance(v, tuple) else (v, None)
    result = dispatch(name, x, y)
    assert result.path == "Fast"
    torch.testing.assert_close(result.output, reference(x, y))
    assert result.repair is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_holdouts_direct_fast_without_copy():
    x = torch.arange(258, device="cuda", dtype=torch.float32)[::2]
    v = dispatch("holdout_vector", x)
    assert v.path == "Fast" and v.repair is None
    torch.testing.assert_close(v.output, reference(x))
    m = torch.arange(7 * 25, device="cuda", dtype=torch.float32).reshape(7, 25)[:, :22:2]
    z = dispatch("holdout_matrix", m)
    assert z.path == "Fast" and z.repair is None
    torch.testing.assert_close(z.output, reference(m))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("name,layout,copied", [("A", "transpose", ("X",)), ("A2", "transpose", ("X",)), ("B", "strided", ("X",)),
                                                ("D", "x_strided", ("X",)), ("D", "y_strided", ("Y",))])
def test_relayout_rechecks_and_preserves_values(name, layout, copied):
    v = make_case(name, layout, n=129 if name in ("B", "D") else 11)
    x, y = v if isinstance(v, tuple) else (v, None)
    default = dispatch(name, x, y)
    assert default.path == "PyTorch Fallback"
    repaired = dispatch(name, x, y, repair_policy="prefer_repair")
    assert repaired.path == "Relayout+Fast", repaired.to_dict()
    assert repaired.repair.copied == copied
    assert repaired.repair.guard.allowed
    torch.testing.assert_close(repaired.output, reference(x, y))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cheap_rejection_short_circuits_pointer_and_span(monkeypatch):
    x = torch.arange(16, device="cuda", dtype=torch.float16)
    monkeypatch.setattr(torch.Tensor, "data_ptr", lambda self: (_ for _ in ()).throw(AssertionError("pointer read")))
    result = get_plan("B").evaluate({"X": x})
    assert result.status == "False" and result.checks == []


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_real_output_capacity_checked_before_launch():
    x = torch.arange(129, device="cuda", dtype=torch.float32)
    short = torch.empty(128, device="cuda")
    guard = get_plan("B").evaluate({"X": x}, output=short)
    assert not guard.allowed and guard.status == "Unknown"
    with pytest.raises(ValueError):
        get_plan("B").launch({"X": x}, guard)
    aliased = get_plan("B").evaluate({"X": x}, output=x)
    assert aliased.status == "Unknown" and not aliased.allowed


def test_bad_binding_and_source_fail_closed():
    original = SPECS["B"]
    bad_meaning = replace(original.meaning, launch_bindings=(("N", "x.size(0)"), ("B", "64"), ("grid", "ceil(N/B)")))
    bad = compile_guard_plan("B", spec=KernelSpec(original.kernel, original.wrapper, bad_meaning))
    assert bad.status == "Unknown"
    bad_wrapper = compile_guard_plan("B", spec=KernelSpec(original.kernel, run_index_hint, original.meaning))
    assert bad_wrapper.status == "Unknown"


def test_missing_span_and_wrong_store_width_fail_closed(monkeypatch):
    from pact.span import SpanResult
    from pact.access_ir import parse_access_ir
    original = SPECS["B"]
    monkeypatch.setattr("pact.guard_plan.extract_spans", lambda *args: SpanResult("Supported", (), "fixture"))
    missing = compile_guard_plan("B")
    assert missing.status == "Unknown" and "缺失访存 Safety" in missing.reason
    monkeypatch.undo()
    real_parse = parse_access_ir
    def wrong_width(*args, **kwargs):
        ir = real_parse(*args, **kwargs)
        return replace(ir, accesses=tuple(replace(point, element_bytes=8) if point.kind == "store" else point for point in ir.accesses))
    monkeypatch.setattr("pact.guard_plan.parse_access_ir", wrong_width)
    rejected = compile_guard_plan("B", spec=original)
    assert rejected.status == "Unknown"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_pointer_and_index_hints_are_distinct():
    pointer = compile_guard_plan("pointer", spec=KernelSpec(pointer_hint, run_pointer_hint, SEMANTICS["B"]))
    index = compile_guard_plan("index", spec=KernelSpec(index_hint, run_index_hint, SEMANTICS["B"]))
    assert pointer.status == index.status == "Supported"
    assert any(check.stage == "pointer" for check in pointer.checks)
    assert not any(check.stage == "pointer" for check in index.checks)
    aligned = torch.arange(128, device="cuda", dtype=torch.float32)
    misaligned = torch.arange(129, device="cuda", dtype=torch.float32)[1:]
    assert pointer.evaluate({"X": aligned}).allowed
    rejected = pointer.evaluate({"X": misaligned})
    assert rejected.status == "False" and rejected.failed.purpose == "Optimization"
    assert misaligned.is_contiguous()
    assert index.evaluate({"X": misaligned}).allowed
    repaired = try_relayout(pointer, {"X": misaligned}, rejected)
    assert repaired.copied == ("X",)
    assert repaired.tensors["X"].untyped_storage().data_ptr() != misaligned.untyped_storage().data_ptr()
    assert repaired.guard.status in ("True", "False")
    if repaired.guard.allowed:
        assert repaired.tensors["X"].data_ptr() % 16 == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cost_table_cannot_revive_rejected_path():
    x = make_case("A", "transpose")
    plan = get_plan("A")
    key = ("A", shape_bucket(tuple(x.shape)), layout_family({"X": x}))
    entry = CostEntry(key, plan.fingerprint, str(x.dtype), str(x.device), (), (("X", x.data_ptr() % 16),), tuple(x.shape),
                      (("Fast", 0.01), ("PyTorch Fallback", 10.0)), 10)
    result = dispatch("A", x, cost_table=CostTable((entry,)))
    assert result.path == "PyTorch Fallback"
    assert "Fast" not in result.feasible
    assert shape_bucket((127,)) != shape_bucket((128,)) != shape_bucket((129,))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_stale_table_does_not_trigger_copy(monkeypatch):
    x = make_case("A", "transpose")
    monkeypatch.setattr("pact.dispatch.try_relayout", lambda *args: (_ for _ in ()).throw(AssertionError("copy")))
    result = dispatch("A", x, cost_table=CostTable(version="stale"))
    assert result.path == "PyTorch Fallback" and result.repair is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_matching_table_chooses_only_rechecked_repair():
    x = make_case("A", "transpose")
    plan = get_plan("A")
    entry = CostEntry(("A", shape_bucket(tuple(x.shape)), layout_family({"X": x})), plan.fingerprint,
                      str(x.dtype), str(x.device), ("X",), (("X", x.data_ptr() % 16),), tuple(x.shape),
                      (("Relayout+Fast", 1.0), ("PyTorch Fallback", 10.0)), 10)
    result = dispatch("A", x, cost_table=CostTable((entry,)))
    assert result.path == "Relayout+Fast" and result.repair.guard.allowed
    torch.testing.assert_close(result.output, reference(x))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_noisy_cost_row_uses_default_without_copy(monkeypatch):
    x = make_case("A", "transpose")
    plan = get_plan("A")
    entry = CostEntry(("A", shape_bucket(tuple(x.shape)), layout_family({"X": x})), plan.fingerprint,
                      str(x.dtype), str(x.device), ("X",), (("X", x.data_ptr() % 16),), tuple(x.shape),
                      (("Relayout+Fast", 1.0), ("PyTorch Fallback", 10.0)), 10, False)
    monkeypatch.setattr("pact.dispatch.try_relayout", lambda *args: (_ for _ in ()).throw(AssertionError("copy")))
    result = dispatch("A", x, cost_table=CostTable((entry,)))
    assert result.path == "PyTorch Fallback" and result.repair is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cost_scope_rejects_dtype_device_and_repair_mismatch():
    x = make_case("B", "contiguous", n=129)
    plan = get_plan("B")
    base = CostEntry(("B", shape_bucket(tuple(x.shape)), layout_family({"X": x})), plan.fingerprint,
                     str(x.dtype), str(x.device), (), (("X", x.data_ptr() % 16),), tuple(x.shape),
                     (("Fast", 10.0), ("PyTorch Fallback", 1.0)), 10)
    feasible = {"Fast", "PyTorch Fallback"}
    assert CostTable((base,)).select(plan, {"X": x}, feasible)[0] == "PyTorch Fallback"
    for wrong in (replace(base, dtype="torch.float16"), replace(base, device="cuda:1"),
                  replace(base, copied=("X",)), replace(base, exact_shape=(128,))):
        assert CostTable((wrong,)).select(plan, {"X": x}, feasible)[0] == "Fast"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_loaded_report_fingerprint_and_environment(tmp_path):
    import json
    import platform
    import triton
    from pact.cost_model import COST_VERSION, repair_fingerprint
    x = make_case("B", "contiguous", n=129)
    plan = get_plan("B")
    entry = CostEntry(("B", shape_bucket(tuple(x.shape)), layout_family({"X": x})), plan.fingerprint,
                      str(x.dtype), str(x.device), (), (("X", x.data_ptr() % 16),), tuple(x.shape),
                      (("Fast", 1.0), ("PyTorch Fallback", 10.0)), 10)
    data = {"version": COST_VERSION, "repair_fingerprint": repair_fingerprint(),
            "environment": {"python": platform.python_version(), "torch": torch.__version__, "triton": triton.__version__,
                            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name()},
            "table": [entry.to_dict()]}
    path = tmp_path / "cost.json"
    path.write_text(json.dumps(data))
    assert CostTable.from_report(path).select(plan, {"X": x}, {"Fast", "PyTorch Fallback"})[0] == "Fast"
    data["repair_fingerprint"] = "stale"
    path.write_text(json.dumps(data))
    assert CostTable.from_report(path).version == "stale"
    path.write_text("null")
    assert CostTable.from_report(path).version == "stale"
    data["repair_fingerprint"] = repair_fingerprint()
    data["table"][0]["median_us"] = [["Fast", "not-a-number"], ["PyTorch Fallback", 10.0]]
    path.write_text(json.dumps(data))
    assert CostTable.from_report(path).version == "stale"
    assert CostTable.from_report(tmp_path / "missing.json").version == "stale"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_kernel_launch_error_is_not_hidden_by_fallback(monkeypatch):
    class BrokenKernel:
        def __getitem__(self, grid):
            raise RuntimeError("launch failed")
    plan = get_plan("B")
    broken = replace(plan, spec=KernelSpec(BrokenKernel(), plan.spec.wrapper, plan.spec.meaning))
    monkeypatch.setattr("pact.dispatch.get_plan", lambda name: broken)
    with pytest.raises(RuntimeError, match="launch failed"):
        dispatch("B", make_case("B", "contiguous", n=129))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_unknown_dtype_falls_back_without_fast():
    x = make_case("B", "unknown_dtype", n=129)
    result = dispatch("B", x)
    assert result.direct_guard.status == "False" and result.path == "PyTorch Fallback"
    torch.testing.assert_close(result.output, reference(x))


def test_unknown_object_with_cost_table_stays_unknown():
    result = dispatch("B", object(), cost_table=CostTable())
    assert result.path == "Unsupported/Unknown"
    assert result.output is None and result.direct_guard.status == "Unknown"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_empty_cost_table_does_not_read_pointer_after_dtype_rejection(monkeypatch):
    x = make_case("B", "unknown_dtype", n=129)
    monkeypatch.setattr(torch.Tensor, "data_ptr", lambda self: (_ for _ in ()).throw(AssertionError("pointer read")))
    result = dispatch("B", x, cost_table=CostTable())
    assert result.path == "PyTorch Fallback"
