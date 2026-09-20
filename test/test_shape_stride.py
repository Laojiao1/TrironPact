"""第三阶段 shape/stride 规则的结构正例和拒绝边界。"""

from dataclasses import replace

import pytest

from pact.access_ir import parse_access_ir
from pact.call_site import audit_call_site
from pact.candidates import CallBinding, BindingResult
from pact.contract_dsl import TensorMetadata
from pact.semantics import SEMANTICS
from pact.shape_stride import extract_shape_stride
from scenarios.external_add import add_kernel, run_add
from scenarios.kernels import KERNELS, run_fast


def _inputs(name, source=None, meaning=None):
    meaning = meaning or SEMANTICS[name]
    pointers = {item.pointer: item.tensor for item in meaning.inputs}
    pointers[meaning.output_pointer] = "OUT"
    kernel = add_kernel if name == "D" else KERNELS[name]
    ir = parse_access_ir(source or kernel, pointers, {key: 4 for key in pointers})
    bindings = dict(meaning.launch_bindings)
    call = CallBinding(tuple(pointers.items()), tuple((key, value) for key, value in meaning.launch_bindings if key != "grid"), bindings["grid"], meaning.inputs[0].shape_params, 4, "wrapper source")
    site = audit_call_site(run_add if name == "D" else run_fast, kernel.fn.__name__, call)
    assert site.status == "Supported", site.reasons
    return ir, meaning, call, site


@pytest.mark.parametrize("name,count", (("A", 2), ("A2", 1), ("B", 1), ("D", 2)))
def test_structural_candidates_for_existing_kernels(name, count):
    result = extract_shape_stride(*_inputs(name))
    assert result.status == "Supported", result.reason
    assert len(result.candidates) == count
    assert all(item.purpose == "Semantics" and item.evidence == "Statically-Proven" for item in result.candidates)
    assert all(item.condition.kind == "or" for item in result.candidates)
    assert all(item.origin.source_line > 0 and item.origin.access_kind == "load" for item in result.candidates)
    assert all("mod_eq" not in str(item.condition.to_dict() if hasattr(item.condition, "to_dict") else item.condition) for item in result.candidates)


def test_singletons_tail_and_padded_row_have_expected_shadow_values():
    a = extract_shape_stride(*_inputs("A")).candidates
    assert all(item.evaluate({"X": TensorMetadata((1, 11), (14, 1), 0, 0x1000, 4, 56)}) for item in a)
    assert all(item.evaluate({"X": TensorMetadata((7, 1), (1, 3), 0, 0x1000, 4, 28)}) for item in a)
    assert a[0].evaluate({"X": TensorMetadata((7, 11), (14, 1), 0, 0x1000, 4, 392)}) is False
    a2 = extract_shape_stride(*_inputs("A2")).candidates
    assert a2[0].evaluate({"X": TensorMetadata((7, 11), (14, 1), 0, 0x1000, 4, 392)}) is True
    b = extract_shape_stride(*_inputs("B")).candidates
    assert b[0].evaluate({"X": TensorMetadata((129,), (1,), 1, 0x1004, 4, 520)}) is True


def test_renaming_and_add_reordering_preserve_conditions():
    base = """
def sample(X, Y, N, B: tl.constexpr):
    index = tl.program_id(0) * B + tl.arange(0, B)
    mask = index < N
    value = tl.load(X + index, mask=mask, other=0)
    tl.store(Y + index, value + 1, mask=mask)
"""
    variant = base.replace("index", "position").replace("tl.program_id(0) * B + tl.arange(0, B)", "tl.arange(0, B) + B * tl.program_id(0)")
    first = extract_shape_stride(*_inputs("B", base)).candidates
    second = extract_shape_stride(*_inputs("B", variant)).candidates
    assert [item.condition for item in first] == [item.condition for item in second]


def test_bad_mask_binding_and_missing_semantics_stay_unknown():
    source = """
def sample(X, Y, N, B: tl.constexpr):
    idx = tl.program_id(0) * B + tl.arange(0, B)
    mask = idx < N
    value = tl.load(X + idx, mask=mask, other=0)
    tl.store(Y + idx, value + 1, mask=tl.arange(0, B) < N)
"""
    ir, meaning, call, site = _inputs("B", source)
    assert extract_shape_stride(ir, meaning, call, site).status == "Unknown"
    ir, meaning, call, site = _inputs("B")
    assert extract_shape_stride(ir, meaning, replace(call, grid="N"), site).status == "Unknown"
    assert extract_shape_stride(ir, meaning, call, BindingResult("Unknown", ("wrapper 不可信",))).status == "Unknown"
    assert extract_shape_stride(ir, None, call, site).status == "Unknown"
    bad_read = replace(meaning, inputs=(replace(meaning.inputs[0], logical_read="x[j]"),))
    assert extract_shape_stride(ir, bad_read, call, site).status == "Unknown"
    bad_domain = replace(meaning, index_domain="0 <= i <= N")
    assert extract_shape_stride(ir, bad_domain, call, site).status == "Unknown"
    bad_write = replace(meaning, logical_write="out[j] = x[i] + 1")
    assert extract_shape_stride(ir, bad_write, call, site).status == "Unknown"
    a_ir, a_meaning, a_call, _ = _inputs("A")
    wrong_grid_meaning = replace(a_meaning, launch_bindings=tuple((key, "ceil(N/B)" if key == "grid" else value) for key, value in a_meaning.launch_bindings))
    assert extract_shape_stride(a_ir, wrong_grid_meaning, replace(a_call, grid="ceil(N/B)"), BindingResult("Supported", ())).status == "Unknown"
