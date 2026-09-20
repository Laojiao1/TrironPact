"""第三阶段 alignment：区分有效指针与索引提示，保持旧 Fast 边界。"""

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from pact.access_ir import parse_access_ir
from pact.alignment import extract_alignment
from pact.call_site import audit_call_site
from pact.candidates import CallBinding
from pact.contract_dsl import TensorMetadata
from pact.semantics import SEMANTICS
from scenarios.alignment_fixtures import index_hint, pointer_hint, run_index_hint, run_pointer_hint


ROOT = Path(__file__).resolve().parent.parent


def _context(kernel, wrapper):
    meaning = SEMANTICS["B"]
    call = CallBinding((("X", "X"), ("Y", "OUT")), (("N", "x.size(0)"), ("B", "128")), "ceil(N/B)", ("N",), 4, "test/alignment_fixtures.py")
    ir = parse_access_ir(kernel, {"X": "X", "Y": "OUT"}, {"X": 4, "Y": 4})
    site = audit_call_site(wrapper, kernel.fn.__name__, call)
    assert ir.status == site.status == "Supported", (ir.reason, site.reasons)
    return ir, meaning, call, site


def test_pointer_hint_is_byte_address_obligation_without_static_fast_grant():
    ir, meaning, call, site = _context(pointer_hint, run_pointer_hint)
    assert len(ir.hints) == 1
    hint = ir.hints[0]
    assert hint.raw_input == "X" and hint.multiple == 16 and hint.pointer_param == "X"
    assert hint.used_access_lines == (ir.accesses[0].line,)
    result = extract_alignment(ir, meaning, call, site)
    assert result.status == "Supported" and len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.purpose == "Optimization" and candidate.evidence == "Unproven"
    assert candidate.condition.kind == "mod_eq" and candidate.condition.left.kind == "effective_ptr"
    assert candidate.condition.divisor == 16 and candidate.origin.source_line == ir.accesses[0].line
    aligned = TensorMetadata((128,), (1,), 0, 0x1000, 4, 512)
    offset = TensorMetadata((128,), (1,), 1, 0x1004, 4, 516)
    assert candidate.evaluate({"X": aligned}) is True
    assert candidate.evaluate({"X": offset}) is False
    # storage_offset 相同也不能代替真实有效地址；data_ptr 已经包含视图起点。
    assert candidate.evaluate({"X": replace(offset, effective_ptr=0x1010)}) is True
    assert candidate.evidence == "Unproven"
    assert "storage_offset" in candidate.reason
    unknown_width = parse_access_ir(pointer_hint, {"X": "X", "Y": "OUT"})
    assert extract_alignment(unknown_width, meaning, call, site).status == "Unknown"


def test_index_hint_has_no_pointer_alignment_claim():
    ir, meaning, call, site = _context(index_hint, run_index_hint)
    hint = ir.hints[0]
    assert hint.pointer_param is None and hint.multiple == 16
    assert set(hint.used_access_lines) == {point.line for point in ir.accesses}
    result = extract_alignment(ir, meaning, call, site)
    assert result.status == "Supported" and len(result.candidates) == 1
    candidate = result.candidates[0]
    assert candidate.condition.left.kind == "scalar" and candidate.condition.left.scalar == "B"
    assert candidate.evidence == "Statically-Proven"
    offset = TensorMetadata((128,), (1,), 1, 0x1004, 4, 516)
    assert candidate.evaluate({"X": offset}, {"B": 128}) is True
    assert candidate.evaluate({"X": offset}, {"B": 8}) is False
    assert "不推出任何 Tensor" in candidate.reason


def test_uninterpreted_hint_and_bad_bindings_do_not_prove_alignment():
    ir, meaning, call, site = _context(index_hint, run_index_hint)
    source = """
def sample(X, Y, N: tl.constexpr, B: tl.constexpr):
    idx = tl.program_id(0) * B + tl.arange(0, B)
    hinted = tl.multiple_of(idx, 16)
    mask = hinted < N
    value = tl.load(X + hinted, mask=mask, other=0)
    tl.store(Y + hinted, value + 1, mask=mask)
"""
    vector = parse_access_ir(source, {"X": "X", "Y": "OUT"}, {"X": 4, "Y": 4})
    assert vector.status == "Supported"
    result = extract_alignment(vector, meaning, call, site)
    assert result.status == "Unknown"
    assert all(item.condition is None and item.evidence == "Unknown" for item in result.candidates)
    assert extract_alignment(ir, meaning, replace(call, grid="N"), site).status == "Unknown"
    assert extract_alignment(ir, None, call, site).status == "Unknown"
    assert parse_access_ir(source.replace("idx, 16", "idx, 12"), {"X": "X", "Y": "OUT"}).status == "Unsupported"
    nested = source.replace("hinted = tl.multiple_of(idx, 16)", "hinted = idx").replace("X + hinted", "X + tl.multiple_of(hinted, 16)")
    assert parse_access_ir(nested, {"X": "X", "Y": "OUT"}).status == "Unsupported"


def test_paired_kernels_execute_correctly_in_isolated_process():
    script = """
import json, sys, torch
from scenarios.alignment_fixtures import run_index_hint, run_pointer_hint
x = torch.arange(129, device='cuda', dtype=torch.float32)
view = x[1:]
assert x.data_ptr() % 16 == 0 and view.data_ptr() % 16 == 4
print(json.dumps({'pointer': bool(torch.equal(run_pointer_hint(x), x + 1)), 'index_offset': bool(torch.equal(run_index_hint(view), view + 1)), 'view_ptr_mod16': view.data_ptr() % 16}))
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout.strip())
    assert payload == {"pointer": True, "index_offset": True, "view_ptr_mod16": 4}
