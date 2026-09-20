"""两种访问结构不同的留出 Kernel 共用 IR 与候选规则。"""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from pact.access_ir import parse_access_ir
from pact.call_site import audit_call_site
from pact.candidates import CallBinding
from pact.contract_dsl import TensorMetadata
from pact.shape_stride import extract_shape_stride
from pact.span import extract_spans
from scenarios.holdouts import MATRIX_MEANING, VECTOR_MEANING, run_strided_vector, run_two_stride_matrix, strided_vector, two_stride_matrix


ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ((strided_vector, run_strided_vector, VECTOR_MEANING, 1), (two_stride_matrix, run_two_stride_matrix, MATRIX_MEANING, 2))


def _context(kernel, wrapper, meaning):
    pointers = {item.pointer: item.tensor for item in meaning.inputs}
    pointers[meaning.output_pointer] = "OUT"
    ir = parse_access_ir(kernel, pointers, {key: 4 for key in pointers})
    bindings = dict(meaning.launch_bindings)
    call = CallBinding(tuple(pointers.items()), tuple((key, value) for key, value in meaning.launch_bindings if key != "grid"), bindings["grid"], meaning.inputs[0].shape_params, 4, "scenarios/holdouts.py")
    site = audit_call_site(wrapper, kernel.fn.__name__, call)
    assert ir.status == site.status == "Supported", (ir.reason, site.reasons)
    return ir, meaning, call, site


@pytest.mark.parametrize("kernel,wrapper,meaning,count", SAMPLES)
def test_holdouts_share_structure_rules_and_have_independent_semantics(kernel, wrapper, meaning, count):
    context = _context(kernel, wrapper, meaning)
    shape = extract_shape_stride(*context)
    span = extract_spans(*context)
    assert shape.status == span.status == "Supported", (shape.reason, span.reason)
    assert len(shape.candidates) == count and len(span.candidates) == 2
    assert all(item.condition is not None and item.evidence == "Statically-Proven" for item in shape.candidates + span.candidates)
    assert all(item.origin.source_file == "scenarios/holdouts.py" for item in shape.candidates + span.candidates)
    assert "stride(0) ==" not in meaning.reference_rule and "stride(1) ==" not in meaning.reference_rule


def test_noncontiguous_holdout_shadow_values_and_rejections():
    vector = _context(*SAMPLES[0][:3])
    shape = extract_shape_stride(*vector)
    span = extract_spans(*vector)
    x = TensorMetadata((129,), (2,), 0, 0x1000, 4, 1032)
    out = TensorMetadata((129,), (1,), 0, 0x2000, 4, 516)
    assert shape.candidates[0].evaluate({"X": x}, {"S": 2}) is True
    assert [item.evaluate({"X": x, "OUT": out}, {"N": 129, "S": 2}) for item in span.candidates] == [True, True]
    assert span.candidates[0].evaluate({"X": replace(x, storage_nbytes=1024)}, {"N": 129, "S": 2}) is False
    matrix = _context(*SAMPLES[1][:3])
    shape2 = extract_shape_stride(*matrix)
    span2 = extract_spans(*matrix)
    xm = TensorMetadata((7, 11), (25, 2), 0, 0x3000, 4, 700)
    outm = TensorMetadata((7, 11), (11, 1), 0, 0x4000, 4, 308)
    assert [item.evaluate({"X": xm}, {"S0": 25, "S1": 2}) for item in shape2.candidates] == [True, True]
    assert [item.evaluate({"X": xm, "OUT": outm}, {"M": 7, "N": 11, "S0": 25, "S1": 2}) for item in span2.candidates] == [True, True]
    assert extract_shape_stride(matrix[0], matrix[1], replace(matrix[2], scalars=tuple((key, "x.stride(0)" if key == "S1" else value) for key, value in matrix[2].scalars)), matrix[3]).status == "Unknown"
    assert extract_spans(matrix[0], matrix[1], replace(matrix[2], output_element_bytes=None), matrix[3]).status == "Unknown"


def test_holdouts_execute_in_isolated_process_without_new_guard_path():
    script = """
import json, torch
from scenarios.holdouts import run_strided_vector, run_two_stride_matrix
from pact.runtime import guard
x = torch.arange(258, device='cuda', dtype=torch.float32)[::2]
y = torch.arange(7*25, device='cuda', dtype=torch.float32).reshape(7,25)[:,:22:2]
print(json.dumps({'vector_correct': bool(torch.equal(run_strided_vector(x), x+1)), 'matrix_correct': bool(torch.equal(run_two_stride_matrix(y), y+1)), 'vector_stride': x.stride(), 'matrix_stride': y.stride(), 'old_guard_vector': guard('holdout_vector', x).allowed, 'old_guard_matrix': guard('holdout_matrix', y).allowed}))
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip())
    assert payload["vector_correct"] and payload["matrix_correct"]
    assert payload["vector_stride"] == [2] and payload["matrix_stride"] == [25, 2]
    assert not payload["old_guard_vector"] and not payload["old_guard_matrix"]
