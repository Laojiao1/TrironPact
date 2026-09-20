"""第三阶段第一块：候选记录、影子求值及受限绑定拒绝。"""

import json
import inspect
from dataclasses import replace

import pytest

from pact.access_ir import Expr, parse_access_ir
from pact.candidates import CallBinding, Candidate, validate_binding
from pact.call_site import audit_call_site
from pact.contract_dsl import Condition, Origin, TensorMetadata, ValueRef
from pact.semantics import SEMANTICS
from scenarios.kernels import KERNELS, run_fast
from scenarios.external_add import add_kernel, run_add


SOURCE = """
def sample(x, out, n, block: tl.constexpr):
    idx = tl.program_id(0) * block + tl.arange(0, block)
    valid = idx < n
    value = tl.load(x + idx, mask=valid, other=0)
    tl.store(out + idx, value + 1, mask=valid)
"""


def _candidate():
    return Candidate(
        Condition("eq", ValueRef("stride", tensor="X", axis=0), ValueRef("constant", value=1)),
        None, "Semantics", "Unproven",
        Origin("fixture.py", 5, "load", "x", "x[i]", (("n", "x.size(0)"),), "比较逻辑与物理索引"),
        "axis_coefficient", ("一维正长度",), (("n", "x.size(0)"),), "尚未完成全域证明",
    )


def test_candidate_roundtrip_and_shadow_evaluation():
    candidate = _candidate()
    assert Candidate.from_dict(json.loads(json.dumps(candidate.to_dict()))) == candidate
    meta = TensorMetadata((7,), (1,), 1, 0x1004, 4, 32)
    assert candidate.evaluate({"X": meta}) is True
    assert candidate.evidence == "Unproven"
    assert candidate.evaluate({}) is None
    obligation = replace(candidate, condition=None, obligation=Expr("param", 2), evidence="Unknown", reason="未解释的地址提示")
    assert Candidate.from_dict(json.loads(json.dumps(obligation.to_dict()))) == obligation
    assert obligation.evaluate({"X": meta}) is None


def test_candidate_rejects_missing_unknown_or_inconsistent_fields():
    data = _candidate().to_dict()
    for mutated in ({k: v for k, v in data.items() if k != "rule"}, {**data, "extra": 1}):
        with pytest.raises((ValueError, TypeError)):
            Candidate.from_dict(mutated)
    with pytest.raises(ValueError):
        Candidate.from_dict({**data, "condition": None})
    with pytest.raises(ValueError):
        Candidate.from_dict({**data, "origin": {**data["origin"], "unknown": 1}})


def test_binding_rejects_wrong_grid_tile_pointer_output_and_missing_semantics():
    meaning = SEMANTICS["B"]
    ir = parse_access_ir(SOURCE, {"x": "X", "out": "OUT"}, {"x": 4, "out": 4})
    # Fixture signature uses lowercase names, so use an independently declared meaning.
    meaning = replace(meaning, inputs=(replace(meaning.inputs[0], pointer="x", shape_params=("n",)),), output_pointer="out", launch_bindings=(("n", "x.size(0)"), ("block", "128"), ("grid", "ceil(n/block)")))
    call = CallBinding((("x", "X"), ("out", "OUT")), (("n", "x.size(0)"), ("block", "128")), "ceil(n/block)", ("n",), 4, "test fixture")
    # Only the explicitly enumerated grid grammar is accepted.
    assert validate_binding(ir, meaning, call).status == "Unknown"
    meaning = replace(meaning, launch_bindings=(("n", "x.size(0)"), ("block", "128"), ("grid", "ceil(N/B)")))
    call = replace(call, grid="ceil(N/B)")
    assert validate_binding(ir, meaning, call).status == "Supported"
    changes = (
        replace(call, grid="n"),
        replace(call, scalars=(("n", "x.size(0)"), ("block", "64"))),
        replace(call, pointers=(("x", "OUT"), ("out", "X"))),
        replace(call, output_shape=("wrong",)),
        replace(call, output_element_bytes=None),
    )
    assert all(validate_binding(ir, meaning, item).status == "Unknown" for item in changes)
    assert validate_binding(ir, None, call).status == "Unknown"
    assert CallBinding.from_dict(json.loads(json.dumps(call.to_dict()))) == call
    with pytest.raises(ValueError):
        CallBinding.from_dict({**call.to_dict(), "unknown": 1})


@pytest.mark.parametrize("name", ("A", "A2", "B", "D"))
def test_current_wrapper_launch_matches_independent_binding(name):
    meaning = SEMANTICS[name]
    pointers = {item.pointer: item.tensor for item in meaning.inputs}
    pointers[meaning.output_pointer] = "OUT"
    kernel = add_kernel if name == "D" else KERNELS[name]
    ir = parse_access_ir(kernel, pointers, {key: 4 for key in pointers})
    bindings = dict(meaning.launch_bindings)
    call = CallBinding(tuple(pointers.items()), tuple((key, value) for key, value in meaning.launch_bindings if key != "grid"), bindings["grid"], meaning.inputs[0].shape_params, 4, "wrapper source")
    assert validate_binding(ir, meaning, call).status == "Supported"
    result = audit_call_site(run_add if name == "D" else run_fast, kernel.fn.__name__, call)
    assert result.status == "Supported", result.reasons


def test_wrapper_source_mismatch_is_not_accepted(monkeypatch):
    meaning = SEMANTICS["D"]
    call = CallBinding((("x_ptr", "X"), ("y_ptr", "Y"), ("output_ptr", "OUT")), tuple((key, value) for key, value in meaning.launch_bindings if key != "grid"), "ceil(n_elements/BLOCK_SIZE)", ("n_elements",), 4, "wrapper source")
    original = inspect.getsource(run_add)
    monkeypatch.setattr(inspect, "getsource", lambda _: original.replace("(x, y, out, x.numel(), block)", "(y, x, out, x.numel(), block)"))
    assert audit_call_site(run_add, "add_kernel", call).status == "Unknown"
    monkeypatch.setattr(inspect, "getsource", lambda _: original.replace("triton.cdiv(x.numel(), block)", "triton.cdiv(x.numel(), block + 1)"))
    assert audit_call_site(run_add, "add_kernel", call).status == "Unknown"
    monkeypatch.setattr(inspect, "getsource", lambda _: original.replace("torch.empty_like(x, memory_format", "torch.empty_like(y, memory_format"))
    assert audit_call_site(run_add, "add_kernel", call).status == "Unknown"
    monkeypatch.setattr(inspect, "getsource", lambda _: original.replace("    return out", "    return y"))
    assert audit_call_site(run_add, "add_kernel", call).status == "Unknown"
    monkeypatch.setattr(inspect, "getsource", lambda _: original.replace("    block = 128", "    block = 128\n    block = 64"))
    assert audit_call_site(run_add, "add_kernel", call).status == "Unknown"
