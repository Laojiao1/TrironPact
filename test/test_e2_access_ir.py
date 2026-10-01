"""e2 Access IR 的通用语法与 fail-closed 边界。"""

from pact.e2_access import parse_extended_access_ir


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


def test_extended_ir_supports_2d_grid_multiple_accesses_and_and_mask():
    result = parse_extended_access_ir(
        TWO_DIMENSIONAL,
        {"x": "X", "y": "Y", "out": "OUT"},
        {"x": 4, "y": 4, "out": 4},
        {"block": 128},
    )
    assert result.status == "Supported", result.reason
    assert len(result.accesses) == 3
    assert {point.kind for point in result.accesses} == {"load", "store"}
    assert any("program_id(1)" in point.index_calls for point in result.accesses)
    assert result.accesses[0].mask.op == "and"


POINTER_UPDATE = """
def kernel(x, out, stride, n, enabled: tl.constexpr, block: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    x += row * stride
    out += row * stride
    col = tl.arange(0, block)
    mask = col < n
    if enabled:
        value = tl.load(x + col, mask=mask)
    else:
        value = 0
    tl.store(out + col, value, mask=mask)
"""


def test_extended_ir_uses_bound_static_branch_and_pointer_updates():
    result = parse_extended_access_ir(
        POINTER_UPDATE,
        {"x": "X", "out": "OUT"},
        {"x": 4, "out": 4},
        {"enabled": True, "block": 64},
    )
    assert result.status == "Supported", result.reason
    assert len(result.accesses) == 2
    assert all("mul(" in point.offset.key() for point in result.accesses)
    missing = parse_extended_access_ir(
        POINTER_UPDATE,
        {"x": "X", "out": "OUT"},
        {"x": 4, "out": 4},
        {"block": 64},
    )
    assert missing.status == "Unknown"


def test_unknown_branch_without_direct_memory_still_fails_closed():
    source = """
def kernel(x, out, stride, choose, block: tl.constexpr):
    if choose:
        x += stride
    lane = tl.arange(0, block)
    value = tl.load(x + lane)
    tl.store(out + lane, value)
"""
    result = parse_extended_access_ir(
        source,
        {"x": "X", "out": "OUT"},
        {"x": 4, "out": 4},
        {"block": 32},
    )
    assert result.status == "Unknown"


MULTI_STORE = """
def kernel(x, left, right, n: tl.constexpr):
    lane = tl.arange(0, n)
    value = tl.load(x + lane)
    tl.store(left + lane, value)
    tl.store(right + lane, value)
"""


def test_extended_ir_supports_multiple_stores_and_unmasked_exact_domain():
    result = parse_extended_access_ir(
        MULTI_STORE,
        {"x": "X", "left": "LEFT", "right": "RIGHT"},
        {"x": 4, "left": 4, "right": 4},
        {"n": 32},
    )
    assert result.status == "Supported", result.reason
    assert [point.kind for point in result.accesses] == ["load", "store", "store"]
    assert all(point.mask.op == "true" for point in result.accesses)


def test_extended_ir_rejects_dynamic_control_flow_indirect_index_and_alias():
    dynamic = TWO_DIMENSIONAL.replace(
        "    a = tl.load(x + row * sx + col, mask=valid)",
        "    if rows > 1:\n        a = tl.load(x + row * sx + col, mask=valid)",
    )
    assert parse_extended_access_ir(dynamic, {"x": "X", "y": "Y", "out": "OUT"}, {"x": 4, "y": 4, "out": 4}, {"block": 128}).status == "Unknown"
    indirect = TWO_DIMENSIONAL.replace("x + row * sx + col", "x + tl.load(y + col)")
    assert parse_extended_access_ir(indirect, {"x": "X", "y": "Y", "out": "OUT"}, {"x": 4, "y": 4, "out": 4}, {"block": 128}).status != "Supported"
    indexed = TWO_DIMENSIONAL.replace("row * sx + col", "row * sx + col[tl.load(y + col)]")
    assert parse_extended_access_ir(indexed, {"x": "X", "y": "Y", "out": "OUT"}, {"x": 4, "y": 4, "out": 4}, {"block": 128}).status == "Unsupported"
    assert parse_extended_access_ir(TWO_DIMENSIONAL, {"x": "X", "y": "X", "out": "OUT"}, {"x": 4, "y": 4, "out": 4}, {"block": 128}).status == "Unknown"


def test_address_cast_where_axis_and_loop_boundaries_are_explicit():
    integer_cast = TWO_DIMENSIONAL.replace("row = tl.program_id(0)", "row = tl.program_id(0).to(tl.int64)")
    assert parse_extended_access_ir(integer_cast, {"x": "X", "y": "Y", "out": "OUT"}, {"x": 4, "y": 4, "out": 4}, {"block": 128}).status == "Supported"
    float_cast = integer_cast.replace("tl.int64", "tl.float32")
    assert parse_extended_access_ir(float_cast, {"x": "X", "y": "Y", "out": "OUT"}, {"x": 4, "y": 4, "out": 4}, {"block": 128}).status != "Supported"
    axis_two = TWO_DIMENSIONAL.replace("tl.program_id(1)", "tl.program_id(2)")
    assert parse_extended_access_ir(axis_two, {"x": "X", "y": "Y", "out": "OUT"}, {"x": 4, "y": 4, "out": 4}, {"block": 128}).status == "Unsupported"
    loop = TWO_DIMENSIONAL.replace("    a = tl.load", "    for _ in range(1):\n        pass\n    a = tl.load")
    assert parse_extended_access_ir(loop, {"x": "X", "y": "Y", "out": "OUT"}, {"x": 4, "y": 4, "out": 4}, {"block": 128}).status == "Unsupported"
    selected = """
def kernel(x, out, choose: tl.constexpr, n: tl.constexpr):
    lane = tl.arange(0, n)
    offset = tl.where(choose, lane, lane + n)
    value = tl.load(x + offset)
    tl.store(out + lane, value)
"""
    supported = parse_extended_access_ir(selected, {"x": "X", "out": "OUT"}, {"x": 4, "out": 4}, {"choose": True, "n": 32})
    assert supported.status == "Supported"
    assert parse_extended_access_ir(selected, {"x": "X", "out": "OUT"}, {"x": 4, "out": 4}, {"n": 32}).status == "Unknown"
