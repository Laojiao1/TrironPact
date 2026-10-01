"""e2 有界 span 与逐访问点候选。"""

from pact.e2_access import parse_extended_access_ir
from pact.e2_contracts import FrozenTensor, enabled_span, extract_bounded_contracts

TWO_DIMENSIONAL = """
def kernel(x, y, out, rows, cols, sx, sy, block: tl.constexpr):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    col = tile * block + tl.arange(0, block)
    valid = (row < rows) & (col < cols)
    a = tl.load(x + row * sx + col, mask=valid)
    b = tl.load(y + col, mask=col < cols)
    tl.store(out + row * sy + col, a + b, mask=valid)
"""


def test_enabled_span_honors_grid_and_and_mask():
    ir = parse_extended_access_ir(
        TWO_DIMENSIONAL,
        {"x": "x", "y": "y", "out": "out"},
        {"x": 4, "y": 4, "out": 4},
        {"block": 8},
    )
    bindings = (("rows", "3"), ("cols", "10"), ("sx", "10"), ("sy", "10"), ("block", "8"))
    tensors = {
        "x": FrozenTensor("x", (3, 10), (10, 1), 4, "float32"),
        "y": FrozenTensor("y", (10,), (1,), 4, "float32"),
        "out": FrozenTensor("out", (3, 10), (10, 1), 4, "float32"),
    }
    result = extract_bounded_contracts(ir, tensors, bindings, (3, 2), "out[row,col] = x[row,col] + y[col]")
    assert result.status == "Supported", result.reason
    assert result.spans == ((0, 29), (0, 9), (0, 29))
    assert len(result.candidates) == 6
    assert all(candidate.binding[-1] == ("grid", "(3, 2)") for candidate in result.candidates)


def test_span_rejects_out_of_bounds_and_missing_binding():
    ir = parse_extended_access_ir(
        TWO_DIMENSIONAL,
        {"x": "x", "y": "y", "out": "out"},
        {"x": 4, "y": 4, "out": 4},
        {"block": 8},
    )
    tensors = {
        "x": FrozenTensor("x", (3, 10), (10, 1), 4, "float32"),
        "y": FrozenTensor("y", (9,), (1,), 4, "float32"),
        "out": FrozenTensor("out", (3, 10), (10, 1), 4, "float32"),
    }
    bindings = (("rows", "3"), ("cols", "10"), ("sx", "10"), ("sy", "10"), ("block", "8"))
    assert extract_bounded_contracts(ir, tensors, bindings, (3, 2), "reference").status == "Unsupported"
    assert enabled_span(ir.accesses[0], ir.signature, {"rows": 3}, (3, 2)) is None
