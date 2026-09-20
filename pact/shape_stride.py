"""由受限 Access IR 与独立逻辑索引提取 shape/stride 影子候选。"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from pact.access_ir import Expr, ParseResult
from pact.analysis import _binding_check, _shape_binding
from pact.candidates import BindingResult, CallBinding, Candidate, validate_binding
from pact.contract_dsl import Condition, Origin, ValueRef
from pact.semantics import OperatorMeaning


@dataclass(frozen=True)
class ShapeResult:
    status: str
    candidates: tuple[Candidate, ...]
    reason: str


def _combine(op: str, *parts: Expr) -> Expr:
    flat = tuple(sub for part in parts for sub in (part.args if part.op == op else (part,)))
    return Expr(op, args=tuple(sorted(flat, key=Expr.key)))


def _param(ir: ParseResult, name: str) -> Expr:
    return Expr("param", ir.signature.index(name))


def _lane(ir: ParseResult, block: str) -> Expr:
    return Expr("lane", args=(_param(ir, block),))


def _indexed_read(text: str, axes: tuple[str, ...]) -> bool:
    try:
        node = ast.parse(text, mode="eval").body
        if not isinstance(node, ast.Subscript) or not isinstance(node.value, ast.Name):
            return False
        indices = node.slice.elts if isinstance(node.slice, ast.Tuple) else (node.slice,)
        return tuple(index.id for index in indices if isinstance(index, ast.Name)) == axes and len(indices) == len(axes)
    except SyntaxError:
        return False


def _write_axes(meaning: OperatorMeaning) -> bool:
    try:
        target = ast.parse(meaning.logical_write.split("=", 1)[0].strip(), mode="eval").body
        if not isinstance(target, ast.Subscript) or not isinstance(target.value, ast.Name) or target.value.id != "out":
            return False
        indices = target.slice.elts if isinstance(target.slice, ast.Tuple) else (target.slice,)
        return len(indices) == len(meaning.axes) and tuple(index.id for index in indices if isinstance(index, ast.Name)) == meaning.axes
    except (SyntaxError, IndexError):
        return False


def _unknown(ir: ParseResult, meaning: OperatorMeaning, reason: str) -> ShapeResult:
    records = []
    for point in ir.accesses:
        if point.kind != "load":
            continue
        item = next((item for item in meaning.inputs if item.tensor == point.tensor), None)
        if item is None:
            continue
        origin = Origin(point.source_file, point.line, "load", point.pointer_param, item.logical_read, meaning.launch_bindings, reason)
        records.append(Candidate(None, point.offset, "Semantics", "Unknown", origin, "compare_logical_address", meaning.assumptions or ("未确认支持域",), meaning.launch_bindings, reason))
    return ShapeResult("Unknown", tuple(records), reason)


def _clause(point, item, meaning, axis: int, expected: ValueRef, reason: str) -> Candidate:
    stride = ValueRef("stride", tensor=item.tensor, axis=axis)
    size = ValueRef("size", tensor=item.tensor, axis=axis)
    one = ValueRef("constant", value=1)
    condition = Condition("or", parts=(Condition("eq", size, one), Condition("eq", stride, expected)))
    origin = Origin(point.source_file, point.line, "load", point.pointer_param, item.logical_read, meaning.launch_bindings, reason)
    return Candidate(condition, None, "Semantics", "Statically-Proven", origin, "compare_logical_address", meaning.assumptions, meaning.launch_bindings, reason)


def _bound_clause(point, item, meaning, axis: int, scalar: str, reason: str) -> Candidate:
    condition = Condition("eq", ValueRef("stride", tensor=item.tensor, axis=axis), ValueRef("scalar", scalar=scalar))
    origin = Origin(point.source_file, point.line, "load", point.pointer_param, item.logical_read, meaning.launch_bindings, reason)
    return Candidate(condition, None, "Semantics", "Statically-Proven", origin, "compare_logical_address", meaning.assumptions, meaning.launch_bindings, reason)


def extract_shape_stride(ir: ParseResult, meaning: OperatorMeaning | None, call: CallBinding, call_site: BindingResult) -> ShapeResult:
    """只支持单轴线性 Tile 或逐行 Tile；静态等级限定在声明语义和调用域。"""
    if meaning is None:
        return ShapeResult("Unknown", (), "缺少独立算子语义")
    checked = validate_binding(ir, meaning, call)
    if checked.status != "Supported" or call_site.status != "Supported":
        return _unknown(ir, meaning, "; ".join(checked.reasons + call_site.reasons) or "调用绑定未核对")
    reason = _binding_check(ir, meaning) or _shape_binding(ir, meaning)
    if reason:
        return _unknown(ir, meaning, reason)
    if any(not _indexed_read(item.logical_read, meaning.axes) for item in meaning.inputs):
        return _unknown(ir, meaning, "独立逻辑读取不是受支持的逐轴索引")
    shape = meaning.inputs[0].shape_params
    if len(shape) != meaning.ndim or any(item.shape_params != shape for item in meaning.inputs) or not all(name in ir.signature for name in shape):
        return _unknown(ir, meaning, "逻辑轴与形状参数无法对应")
    expected_domain = "，".join(f"0 <= {axis} < {param}" for axis, param in zip(meaning.axes, shape))
    if meaning.index_domain != expected_domain or not _write_axes(meaning):
        return _unknown(ir, meaning, "独立索引域或逻辑输出轴无法与形状参数对应")
    store = next(point for point in ir.accesses if point.kind == "store")
    if any(point.mask != store.mask for point in ir.accesses):
        return _unknown(ir, meaning, "load/store mask 不一致，不能证明相同索引域")
    block = next((name for name, value in call.scalars if name in store.constexpr and (value.isdigit() or value.startswith("next_power_of_2("))), None)
    if block is None:
        return _unknown(ir, meaning, "Tile 参数不是受支持的 constexpr 绑定")
    lane, pid = _lane(ir, block), Expr("pid", 0)
    total = _param(ir, shape[0]) if len(shape) == 1 else _combine("mul", *(_param(ir, name) for name in shape))
    linear = _combine("add", _combine("mul", pid, _param(ir, block)), lane)
    flat = store.offset == linear and store.mask == Expr("lt", args=(linear, total))
    row = len(shape) == 2 and store.offset == _combine("add", _combine("mul", pid, _param(ir, shape[1])), lane) and store.mask == Expr("lt", args=(lane, _param(ir, shape[1])))
    expected_flat_grid = f"ceil({'*'.join(shape)}/{block})"
    if flat and call.grid != expected_flat_grid:
        return _unknown(ir, meaning, "扁平 Tile 的 grid 无法证明覆盖全部逻辑索引")
    if row and call.grid != shape[0]:
        return _unknown(ir, meaning, "逐行 Tile 的 grid 无法证明覆盖全部行")
    if not (flat or row):
        return _unknown(ir, meaning, "输出写入与 mask 无法映射到受支持的逻辑索引域")
    candidates = []
    for point in (item for item in ir.accesses if item.kind == "load"):
        item = meaning.input_for(point.tensor)
        if flat:
            if point.offset != linear:
                source = ast.parse(item.logical_read, mode="eval").body.value.id
                stride_param = next((name for name, value in call.scalars if value == f"{source}.stride(0)"), None)
                if meaning.ndim != 1 or stride_param is None or point.offset != _combine("mul", linear, _param(ir, stride_param)):
                    return _unknown(ir, meaning, "输入地址不能与扁平逻辑索引或已绑定的一维步长对应")
                candidates.append(_bound_clause(point, item, meaning, 0, stride_param, "一维读取地址为 i*已核对的输入 stride(0)，按逻辑索引逐元素读取"))
                continue
            if meaning.ndim == 1:
                candidates.append(_clause(point, item, meaning, 0, ValueRef("constant", value=1), "mask 限定 0<=i<N；相邻逻辑元素地址差为 1，单元素时步长无作用"))
            elif meaning.ndim == 2:
                candidates.append(_clause(point, item, meaning, 0, ValueRef("size", tensor=item.tensor, axis=1), "扁平地址按列数 N 换行；单行时行步长无作用"))
                candidates.append(_clause(point, item, meaning, 1, ValueRef("constant", value=1), "同一行相邻列地址差为 1；单列时列步长无作用"))
            else:
                return _unknown(ir, meaning, "扁平 Tile 只支持一维或二维语义")
        else:
            source = ast.parse(item.logical_read, mode="eval").body.value.id
            bound = dict(call.scalars)
            row_stride = next((name for name, value in bound.items() if value == f"{source}.stride(0)"), None)
            col_stride = next((name for name, value in bound.items() if value == f"{source}.stride(1)"), None)
            col_term = _combine("mul", lane, _param(ir, col_stride)) if col_stride is not None else lane
            if row_stride is None or point.offset != _combine("add", _combine("mul", pid, _param(ir, row_stride)), col_term):
                return _unknown(ir, meaning, "逐行读取的行步长未绑定输入 stride(0) 或列系数未知")
            if bound.get(block) != f"next_power_of_2({shape[1]})":
                return _unknown(ir, meaning, "无法证明逐行 Tile 宽度覆盖列维度")
            if col_stride is None:
                candidates.append(_clause(point, item, meaning, 1, ValueRef("constant", value=1), "行系数由已核对的 stride(0) 提供；列地址差为 1，单列时列步长无作用"))
            else:
                candidates.append(_bound_clause(point, item, meaning, 0, row_stride, "逐行读取地址的行系数来自已核对的输入 stride(0)"))
                candidates.append(_bound_clause(point, item, meaning, 1, col_stride, "逐列读取地址的列系数来自已核对的输入 stride(1)"))
    return ShapeResult("Supported", tuple(candidates), "受支持的输出逻辑索引、mask、grid 与输入地址逐轴比较完成；不授予 Fast 资格")
