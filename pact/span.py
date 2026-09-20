"""从已启用的单轴/逐行 Tile 逐访存求元素偏移范围；只作离线候选。"""

from __future__ import annotations

from dataclasses import dataclass

from pact.access_ir import Expr, ParseResult
from pact.analysis import _binding_check, _shape_binding
from pact.candidates import BindingResult, CallBinding, Candidate, validate_binding
from pact.contract_dsl import AccessSpan, BoundExpr, Condition, Origin
from pact.semantics import OperatorMeaning
from pact.shape_stride import _write_axes


@dataclass(frozen=True)
class SpanResult:
    status: str
    candidates: tuple[Candidate, ...]
    reason: str


def _c(value: int) -> BoundExpr:
    return BoundExpr("constant", value=value)


def _s(name: str) -> BoundExpr:
    return BoundExpr("scalar", scalar=name)


def _op(kind: str, left: BoundExpr, right: BoundExpr) -> BoundExpr:
    return BoundExpr(kind, parts=(left, right))


def _expr(op: str, *parts: Expr) -> Expr:
    flat = tuple(sub for part in parts for sub in (part.args if part.op == op else (part,)))
    return Expr(op, args=tuple(sorted(flat, key=Expr.key)))


def _param(ir: ParseResult, name: str) -> Expr:
    return Expr("param", ir.signature.index(name))


def _terms(expr: Expr) -> tuple[Expr, ...]:
    return expr.args if expr.op == "add" else (expr,)


def _constant_shift(offset: Expr, base: Expr) -> int | None:
    terms = list(_terms(offset))
    for term in _terms(base):
        if term not in terms:
            return None
        terms.remove(term)
    return sum(term.value for term in terms) if all(term.op == "const" and type(term.value) is int for term in terms) else None


def _row_parts(offset: Expr, lane: Expr, ir: ParseResult) -> tuple[str, str | None, int] | None:
    terms = list(_terms(offset))
    col_coefficient = None
    if lane in terms:
        terms.remove(lane)
    else:
        col = next((term for term in terms if term.op == "mul" and lane in term.args and len(term.args) == 2), None)
        if col is None:
            return None
        terms.remove(col)
        coefficient = next(item for item in col.args if item != lane)
        if coefficient.op != "param" or type(coefficient.value) is not int or coefficient.value >= len(ir.signature):
            return None
        col_coefficient = ir.signature[coefficient.value]
    row = next((term for term in terms if term.op == "mul" and Expr("pid", 0) in term.args and len(term.args) == 2), None)
    if row is None:
        return None
    terms.remove(row)
    coefficient = next(item for item in row.args if item != Expr("pid", 0))
    if coefficient.op != "param" or type(coefficient.value) is not int or coefficient.value >= len(ir.signature):
        return None
    if not all(term.op == "const" and type(term.value) is int for term in terms):
        return None
    return ir.signature[coefficient.value], col_coefficient, sum(term.value for term in terms)


def _origin(point, meaning: OperatorMeaning, reason: str) -> Origin:
    semantic = meaning.logical_write if point.kind == "store" else next((item.logical_read for item in meaning.inputs if item.tensor == point.tensor), "未匹配输入")
    return Origin(point.source_file, point.line, point.kind, point.pointer_param, semantic, meaning.launch_bindings, reason)


def _unknown(ir: ParseResult, meaning: OperatorMeaning, reason: str) -> SpanResult:
    records = tuple(Candidate(None, point.offset, "Safety", "Unknown", _origin(point, meaning, reason), "enabled_access_range", meaning.assumptions or ("未确认支持域",), meaning.launch_bindings, reason) for point in ir.accesses)
    return SpanResult("Unknown", records, reason)


def extract_spans(ir: ParseResult, meaning: OperatorMeaning | None, call: CallBinding, call_site: BindingResult) -> SpanResult:
    if meaning is None:
        return SpanResult("Unknown", (), "缺少独立算子语义")
    checked = validate_binding(ir, meaning, call)
    if checked.status != "Supported" or call_site.status != "Supported":
        return _unknown(ir, meaning, "; ".join(checked.reasons + call_site.reasons) or "调用绑定未核对")
    semantic_reason = _binding_check(ir, meaning) or _shape_binding(ir, meaning)
    if semantic_reason:
        return _unknown(ir, meaning, semantic_reason)
    shape = meaning.inputs[0].shape_params
    expected_domain = "，".join(f"0 <= {axis} < {param}" for axis, param in zip(meaning.axes, shape))
    if len(shape) != meaning.ndim or meaning.index_domain != expected_domain or not _write_axes(meaning) or not all(name in ir.signature for name in shape):
        return _unknown(ir, meaning, "独立索引域或输出轴无法与形状参数对应")
    store = next((point for point in ir.accesses if point.kind == "store"), None)
    if store is None or any(point.mask != store.mask for point in ir.accesses):
        return _unknown(ir, meaning, "load/store mask 不一致或输出缺失")
    block = next((name for name, value in call.scalars if name in store.constexpr and (value.isdigit() or value.startswith("next_power_of_2("))), None)
    if block is None:
        return _unknown(ir, meaning, "Tile 参数未核对")
    lane = Expr("lane", args=(_param(ir, block),))
    pid = Expr("pid", 0)
    linear = _expr("add", _expr("mul", pid, _param(ir, block)), lane)
    total = _param(ir, shape[0]) if len(shape) == 1 else _expr("mul", *(_param(ir, name) for name in shape))
    flat = store.offset == linear and store.mask == Expr("lt", args=(linear, total)) and call.grid == f"ceil({'*'.join(shape)}/{block})"
    row = len(shape) == 2 and store.offset == _expr("add", _expr("mul", pid, _param(ir, shape[1])), lane) and store.mask == Expr("lt", args=(lane, _param(ir, shape[1]))) and call.grid == shape[0] and dict(call.scalars).get(block) == f"next_power_of_2({shape[1]})"
    if not flat and not row:
        return _unknown(ir, meaning, "grid、Tile、mask 和输出地址不能证明完整启用域")
    records = []
    total_bound = _s(shape[0]) if len(shape) == 1 else _op("mul", _s(shape[0]), _s(shape[1]))
    for point in ir.accesses:
        if point.element_bytes is None or point.element_bytes < 1:
            return _unknown(ir, meaning, "访存元素宽度未知")
        if flat:
            shift = _constant_shift(point.offset, linear)
            if shift is not None:
                lower = _c(shift)
                upper = _op("add", _op("sub", total_bound, _c(1)), _c(shift))
                explanation = f"mask 限定 0<=idx<{'*'.join(shape)}；该访存偏移为 idx+{shift}，实际启用范围为 [{shift}, total-1+{shift}] 元素"
            else:
                stride = next((name for name, value in call.scalars if value.endswith(".stride(0)") and point.offset == _expr("mul", linear, _param(ir, name))), None)
                if len(shape) != 1 or stride is None:
                    return _unknown(ir, meaning, "扁平 mask 下该访存地址不是固定平移或已绑定的非负一维步长")
                lower = _c(0)
                upper = _op("mul", _op("sub", total_bound, _c(1)), _s(stride))
                explanation = f"mask 限定 0<=idx<{shape[0]}；读取偏移为 idx*{stride}，非负步长时实际启用范围为 [0, ({shape[0]}-1)*{stride}] 元素"
        else:
            parts = _row_parts(point.offset, lane, ir)
            if parts is None:
                return _unknown(ir, meaning, "逐行访存地址不是 pid*非负行距+lane+固定平移")
            coefficient, col_coefficient, shift = parts
            bound = dict(call.scalars).get(coefficient, "")
            col_bound = dict(call.scalars).get(col_coefficient, "") if col_coefficient else ""
            if coefficient not in shape and not (bound.endswith(".stride(0)") or bound.isdigit()):
                return _unknown(ir, meaning, "逐行系数没有可核对的非负标量来源")
            if col_coefficient is not None and not (col_bound.endswith(".stride(1)") or col_bound.isdigit()):
                return _unknown(ir, meaning, "逐列系数没有可核对的非负标量来源")
            lower = _c(shift)
            col_max = _op("mul", _op("sub", _s(shape[1]), _c(1)), _s(col_coefficient)) if col_coefficient else _op("sub", _s(shape[1]), _c(1))
            upper = _op("add", _op("add", _op("mul", _op("sub", _s(shape[0]), _c(1)), _s(coefficient)), col_max), _c(shift))
            col_text = f"({shape[1]}-1)*{col_coefficient}" if col_coefficient else f"{shape[1]}-1"
            explanation = f"grid 覆盖 0<=pid<{shape[0]}，mask 限定 0<=lane<{shape[1]}；行列系数需非负，实际启用范围为 [{shift}, ({shape[0]}-1)*{coefficient}+{col_text}+{shift}] 元素"
        condition = Condition("access_span", span=AccessSpan(point.tensor, lower, upper, point.element_bytes))
        explanation += "；storage_offset 按元素计，端点乘元素宽度后与 storage_nbytes 比较；data_ptr 不重复叠加 offset"
        records.append(Candidate(condition, None, "Safety", "Statically-Proven", _origin(point, meaning, explanation), "enabled_access_range", meaning.assumptions + ("形状正长度", "行系数非负"), meaning.launch_bindings, explanation))
    return SpanResult("Supported", tuple(records), "每个启用的 load/store 均有独立访问范围义务；不授予 Fast 资格")
