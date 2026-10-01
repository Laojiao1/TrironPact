"""冻结后的 e2 规则不变量；本文件不参与规则设计或留出调试。"""

from pact.e2_access import parse_extended_access_ir


BASE = """
def kernel(x, out, rows, stride, block: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, block)
    valid = col < rows
    value = tl.load(x + row * stride + col, mask=valid)
    tl.store(out + row * stride + col, value, mask=valid)
"""


def _normalized(source: str, pointers: dict[str, str]):
    result = parse_extended_access_ir(source, pointers, {name: 4 for name in pointers}, {"block": 32, "tile": 32})
    assert result.status == "Supported", result.reason
    return tuple((point.kind, point.tensor, point.offset, point.mask) for point in result.accesses)


def test_parameter_renaming_and_commutative_rewrite_preserve_access_ir():
    renamed = BASE.replace("kernel(x, out, rows, stride, block", "kernel(a, z, n, step, tile")
    renamed = renamed.replace("block)", "tile)").replace("rows", "n").replace("stride", "step")
    renamed = renamed.replace("tl.load(x + row * step + col", "tl.load(col + row * step + a")
    renamed = renamed.replace("tl.store(out + row * step + col", "tl.store(col + row * step + z")
    assert _normalized(BASE, {"x": "X", "out": "OUT"}) == _normalized(renamed, {"a": "X", "z": "OUT"})
