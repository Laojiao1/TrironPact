"""第三阶段逐访存范围：mask、storage 坐标、输出与拒绝边界。"""

import json
from dataclasses import replace

import pytest

from pact.access_ir import parse_access_ir
from pact.call_site import audit_call_site
from pact.candidates import CallBinding, Candidate
from pact.contract_dsl import TensorMetadata
from pact.semantics import SEMANTICS
from pact.span import extract_spans
from scenarios.external_add import add_kernel, run_add
from scenarios.kernels import KERNELS, run_fast


def _context(name):
    meaning = SEMANTICS[name]
    pointers = {item.pointer: item.tensor for item in meaning.inputs}
    pointers[meaning.output_pointer] = "OUT"
    kernel = add_kernel if name == "D" else KERNELS[name]
    ir = parse_access_ir(kernel, pointers, {key: 4 for key in pointers})
    bindings = dict(meaning.launch_bindings)
    call = CallBinding(tuple(pointers.items()), tuple((key, value) for key, value in meaning.launch_bindings if key != "grid"), bindings["grid"], meaning.inputs[0].shape_params, 4, "wrapper source")
    site = audit_call_site(run_add if name == "D" else run_fast, kernel.fn.__name__, call)
    assert ir.status == site.status == "Supported"
    return ir, meaning, call, site


@pytest.mark.parametrize("name,count", (("A", 2), ("A2", 2), ("B", 2), ("D", 3)))
def test_each_enabled_access_has_independent_span(name, count):
    ir, meaning, call, site = _context(name)
    result = extract_spans(ir, meaning, call, site)
    assert result.status == "Supported", result.reason
    assert len(result.candidates) == count
    assert [(item.origin.access_kind, item.origin.source_line) for item in result.candidates] == [(point.kind, point.line) for point in ir.accesses]
    assert all(item.condition.kind == "access_span" and item.purpose == "Safety" and item.evidence == "Statically-Proven" for item in result.candidates)
    assert Candidate.from_dict(json.loads(json.dumps(result.candidates[0].to_dict()))) == result.candidates[0]


def test_offset_tail_exact_end_and_one_past_storage():
    result = extract_spans(*_context("B"))
    x = TensorMetadata((129,), (1,), 1, 0x1004, 4, 520)
    out = TensorMetadata((129,), (1,), 0, 0x2000, 4, 516)
    assert [item.evaluate({"X": x, "OUT": out}, {"N": 129, "B": 128}) for item in result.candidates] == [True, True]
    assert result.candidates[0].evaluate({"X": replace(x, storage_nbytes=516)}, {"N": 129}) is False
    assert result.candidates[0].evaluate({"X": replace(x, effective_ptr=0x1010)}, {"N": 129}) is True
    assert result.candidates[1].evaluate({"OUT": replace(out, storage_nbytes=512)}, {"N": 129}) is False
    assert result.candidates[1].evaluate({"X": x}, {"N": 129}) is None
    assert all("storage_offset" in item.reason and "data_ptr" in item.reason for item in result.candidates)


def test_padded_row_and_negative_stride_boundary():
    result = extract_spans(*_context("A2"))
    x = TensorMetadata((7, 11), (14, 1), 0, 0x1000, 4, 392)
    out = TensorMetadata((7, 11), (11, 1), 0, 0x2000, 4, 308)
    meta = {"X": x, "OUT": out}
    scalars = {"M": 7, "N": 11, "S0": 14, "B": 16}
    assert [item.evaluate(meta, scalars) for item in result.candidates] == [True, True]
    assert result.candidates[0].evaluate(meta, {**scalars, "S0": -1}) is None
    assert result.candidates[0].evaluate({"X": replace(x, stride=(-1, 1))}, scalars) is None
    assert result.candidates[0].evaluate({"X": replace(x, storage_nbytes=376)}, scalars) is False


def test_unknown_grid_width_output_and_semantics_are_rejected():
    ir, meaning, call, site = _context("B")
    assert extract_spans(ir, meaning, replace(call, grid="N"), site).status == "Unknown"
    assert extract_spans(ir, meaning, replace(call, output_shape=()), site).status == "Unknown"
    assert extract_spans(ir, meaning, replace(call, output_element_bytes=None), site).status == "Unknown"
    assert extract_spans(parse_access_ir(KERNELS["B"], {"X": "X", "Y": "OUT"}), meaning, call, site).status == "Unknown"
    assert extract_spans(ir, None, call, site).status == "Unknown"
    wrong_domain = replace(meaning, index_domain="0 <= i <= N")
    assert extract_spans(ir, wrong_domain, call, site).status == "Unknown"
