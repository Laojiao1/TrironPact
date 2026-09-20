"""验证统一 AST 访存解析、结构等价与保守拒绝边界。"""

from pact.access_ir import parse_access_ir
from scenarios.external_add import add_kernel
from scenarios.kernels import flat_2d, masked_1d, row_stride_2d


POC = (
    (flat_2d, {"X": "X", "Y": "OUT"}, 2),
    (row_stride_2d, {"X": "X", "Y": "OUT"}, 2),
    (masked_1d, {"X": "X", "Y": "OUT"}, 2),
    (add_kernel, {"x_ptr": "X", "y_ptr": "Y", "output_ptr": "OUT"}, 3),
)


def test_four_poc_kernels_share_one_parser():
    for kernel, bindings, count in POC:
        result = parse_access_ir(kernel, bindings)
        assert result.status == "Supported", result.reason
        assert len(result.accesses) == count
        assert result.store_value_supported
        assert all(access.mask.op == "lt" and access.source_file and access.line > 0 for access in result.accesses)
        assert all(access.element_bytes is None for access in result.accesses)


def test_renamed_variables_and_reordered_addition_have_same_ir():
    original = parse_access_ir(masked_1d, {"X": "X", "Y": "OUT"})
    variant = parse_access_ir("""
def renamed(a, b, length, tile: tl.constexpr):
    lane = tl.arange(0, tile)
    p = tl.program_id(0)
    address = lane + p * tile
    valid = address < length
    data = tl.load(a + address, mask=valid, other=0)
    tl.store(address + b, 1 + data, mask=valid)
""", {"a": "X", "b": "OUT"})
    assert variant.status == "Supported", variant.reason
    assert [item.offset.key() for item in variant.accesses] == [item.offset.key() for item in original.accesses]
    assert [item.mask.key() for item in variant.accesses] == [item.mask.key() for item in original.accesses]
    assert variant.accesses[-1].value == original.accesses[-1].value


def test_unsupported_address_mask_control_flow_and_binding():
    base = """
def kernel(x, out, n, block: tl.constexpr):
    lane = tl.program_id(0) * block + tl.arange(0, block)
    valid = lane < n
    value = tl.load(x + lane, mask=valid, other=0)
    tl.store(out + lane, value + 1, mask=valid)
"""
    bindings = {"x": "X", "out": "OUT"}
    assert parse_access_ir(base, bindings).status == "Supported"
    assert parse_access_ir(base.replace("block: tl.constexpr", "block"), bindings).status == "Unknown"
    assert parse_access_ir(base.replace("mask=valid, other=0", "other=0"), bindings).status == "Unsupported"
    assert parse_access_ir(base.replace("x + lane", "x + lane * lane"), bindings).status == "Unsupported"
    reassigned = base.replace("    valid =", "    lane = lane + 1\n    valid =")
    assert parse_access_ir(reassigned, bindings).status == "Unsupported"
    nested_load = base.replace("value = tl.load(x + lane, mask=valid, other=0)", "value = tl.load(x + lane, mask=valid, other=0) + 1")
    assert parse_access_ir(nested_load, bindings).status == "Unsupported"
    indirect = base.replace("    value =", "    index = tl.load(out + lane, mask=valid, other=0)\n    value =").replace("x + lane, mask", "x + index, mask")
    assert parse_access_ir(indirect, bindings).status == "Unsupported"
    assert parse_access_ir(base.replace("    value =", "    if n > 0:\n        value ="), bindings).status == "Unsupported"
    assert parse_access_ir(base, {"wrong": "X", "out": "OUT"}).status == "Unknown"
    assert parse_access_ir(base, {"x": "X"}).status == "Unknown"
    assert parse_access_ir(None, bindings).status == "Unknown"
    hinted = parse_access_ir(base.replace("    valid =", "    hint = tl.multiple_of(lane, 16)\n    valid ="), bindings)
    assert hinted.status == "Supported" and len(hinted.hints) == 1
    assert hinted.hints[0].raw_input == "lane" and hinted.hints[0].used_access_lines == ()
