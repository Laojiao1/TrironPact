"""从绑定后的 e2 Access IR 生成有界、逐访问点的契约候选。"""

from __future__ import annotations

import ast
import itertools
import math
from dataclasses import dataclass

from pact.access_ir import AccessPoint, Expr, ParseResult
from pact.candidates import Candidate
from pact.contract_dsl import AccessSpan, BoundExpr, Condition, Origin, ValueRef


@dataclass(frozen=True)
class FrozenTensor:
    """冻结调用域中的张量元数据；shape/stride 单位为元素。"""

    name: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    element_bytes: int
    dtype: str

    @property
    def numel(self) -> int:
        return math.prod(self.shape)


@dataclass(frozen=True)
class ContractResult:
    status: str
    candidates: tuple[Candidate, ...]
    spans: tuple[tuple[int, int], ...]
    reason: str


_Variable = tuple[str, tuple[str, ...], int]


def contiguous_stride(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride: list[int] = []
    running = 1
    for size in reversed(shape):
        stride.append(running)
        running *= size
    return tuple(reversed(stride))


def binding_values(bindings: tuple[tuple[str, str], ...], tensors: dict[str, FrozenTensor]) -> dict[str, int]:
    """求值地址需要的整数 wrapper 实参；未知表达式不进入结果。"""

    result: dict[str, int] = {}
    for name, expression in bindings:
        try:
            node = ast.parse(expression, mode="eval").body
        except SyntaxError:
            continue
        value: object | None = None
        if isinstance(node, ast.Constant):
            value = node.value
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            tensor = tensors.get(node.func.value.id)
            if tensor is not None and node.func.attr == "numel" and not node.args:
                value = tensor.numel
            elif tensor is not None and node.func.attr in ("size", "stride") and len(node.args) == 1:
                try:
                    axis = ast.literal_eval(node.args[0])
                except (ValueError, SyntaxError):
                    axis = None
                if type(axis) is int and 0 <= axis < len(tensor.shape):
                    value = tensor.shape[axis] if node.func.attr == "size" else tensor.stride[axis]
        if type(value) in (int, bool):
            result[name] = int(value)
    return result


def _variables(expr: Expr, context: tuple[str, ...] = ()) -> set[_Variable]:
    if expr.op == "broadcast":
        return _variables(expr.args[0], context + (str(expr.value),))
    if expr.op == "pid":
        return {("pid", (), int(expr.value))}
    if expr.op == "lane":
        bound = expr.args[0]
        if bound.op != "const" or type(bound.value) is not int:
            return {("lane_param", context, -1)}
        return {("lane", context, int(bound.value))}
    result: set[_Variable] = set()
    for item in expr.args:
        result.update(_variables(item, context))
    return result


def _eval(expr: Expr, assignment: dict[_Variable, int], params: dict[int, int], context: tuple[str, ...] = ()) -> int | bool:
    if expr.op == "const":
        return int(expr.value)
    if expr.op == "param":
        if int(expr.value) not in params:
            raise ValueError(f"参数位置 {expr.value} 没有整数绑定")
        return params[int(expr.value)]
    if expr.op == "pid":
        return assignment[("pid", (), int(expr.value))]
    if expr.op == "lane":
        bound = _eval(expr.args[0], assignment, params, context)
        return assignment[("lane", context, int(bound))]
    if expr.op == "broadcast":
        return _eval(expr.args[0], assignment, params, context + (str(expr.value),))
    if expr.op == "add":
        return sum(int(_eval(item, assignment, params, context)) for item in expr.args)
    if expr.op == "mul":
        value = 1
        for item in expr.args:
            value *= int(_eval(item, assignment, params, context))
        return value
    if expr.op == "floordiv":
        return int(_eval(expr.args[0], assignment, params, context)) // int(expr.value)
    if expr.op == "lt":
        return int(_eval(expr.args[0], assignment, params, context)) < int(_eval(expr.args[1], assignment, params, context))
    if expr.op == "and":
        return all(bool(_eval(item, assignment, params, context)) for item in expr.args)
    if expr.op == "true":
        return True
    if expr.op == "false":
        return False
    raise ValueError(f"不支持求值的 IR 节点：{expr.op}")


def enabled_span(point: AccessPoint, signature: tuple[str, ...], values: dict[str, int], grid: tuple[int, ...]) -> tuple[int, int] | None:
    """枚举冻结 launch 中启用的坐标，返回元素偏移闭区间。"""

    params = {index: values[name] for index, name in enumerate(signature) if name in values}
    variables = _variables(point.offset) | _variables(point.mask)
    domains: list[tuple[_Variable, range]] = []
    total = 1
    for variable in sorted(variables):
        kind, _, value = variable
        if kind == "pid":
            if value >= len(grid) or grid[value] < 1:
                return None
            domain = range(grid[value])
        elif kind == "lane":
            if value < 1:
                return None
            domain = range(value)
        else:
            return None
        total *= len(domain)
        if total > 2_000_000:
            return None
        domains.append((variable, domain))
    offsets: list[int] = []
    combinations = itertools.product(*(domain for _, domain in domains)) if domains else [()]
    try:
        for combination in combinations:
            assignment = {variable: value for (variable, _), value in zip(domains, combination)}
            if bool(_eval(point.mask, assignment, params)):
                offsets.append(int(_eval(point.offset, assignment, params)))
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    return (min(offsets), max(offsets)) if offsets else (0, -1)


def _eq(kind: str, tensor: str, value: int, axis: int | None = None) -> Condition:
    return Condition("eq", ValueRef(kind, tensor=tensor, axis=axis), ValueRef("constant", value=value))


def _metadata_condition(tensor: FrozenTensor) -> Condition:
    parts = [_eq("storage_offset", tensor.name, 0), _eq("element_bytes", tensor.name, tensor.element_bytes)]
    for axis, (size, stride) in enumerate(zip(tensor.shape, tensor.stride)):
        parts.extend((_eq("size", tensor.name, size, axis), _eq("stride", tensor.name, stride, axis)))
    return Condition("and", parts=tuple(parts))


def extract_bounded_contracts(
    ir: ParseResult,
    tensors: dict[str, FrozenTensor],
    bindings: tuple[tuple[str, str], ...],
    grid: tuple[int, ...],
    semantic_reference: str,
) -> ContractResult:
    """为冻结 wrapper 域生成逐访问点候选。

    Safety span 来自绑定后坐标的穷举静态求值；Semantics 条件只声明物理
    元数据与已审冻结参考一致。普通算术和 reduction 数值语义不在证明范围。
    """

    if ir.status != "Supported":
        return ContractResult(ir.status, (), (), ir.reason)
    values = binding_values(bindings, tensors)
    candidates: list[Candidate] = []
    spans: list[tuple[int, int]] = []
    domain = (
        "冻结 wrapper 绑定与 grid",
        "非空 CUDA Tensor",
        "输入输出 storage 不 alias",
        "只证明访存坐标与物理元数据，不证明数值算术",
    )
    for point in ir.accesses:
        tensor = tensors.get(point.tensor)
        if tensor is None:
            return ContractResult("Unknown", tuple(candidates), tuple(spans), f"访问点 {point.pointer_param} 缺少 Tensor 规格")
        span = enabled_span(point, ir.signature, values, grid)
        if span is None:
            return ContractResult("Unknown", tuple(candidates), tuple(spans), f"访问点 {point.pointer_param}:{point.line} 的启用域无法求值")
        lower, upper = span
        if upper < lower:
            return ContractResult("Unknown", tuple(candidates), tuple(spans), f"访问点 {point.pointer_param}:{point.line} 在冻结域无启用坐标")
        if lower < 0 or upper >= tensor.numel:
            return ContractResult("Unsupported", tuple(candidates), tuple(spans), f"访问点 {point.pointer_param}:{point.line} 的偏移 [{lower},{upper}] 超出冻结 Tensor")
        spans.append(span)
        binding = bindings + (("grid", str(tuple(grid))),)
        explanation = (
            f"冻结绑定下启用元素偏移为 [{lower},{upper}]；端点单位为元素，"
            "AccessSpan 乘 element_bytes 后与 storage_nbytes 比较，data_ptr 已含视图偏移。"
        )
        origin = Origin(point.source_file, point.line, point.kind, point.pointer_param, semantic_reference, binding, explanation)
        safety = Condition(
            "access_span",
            span=AccessSpan(tensor.name, BoundExpr("constant", value=lower), BoundExpr("constant", value=upper), tensor.element_bytes),
        )
        candidates.append(Candidate(safety, None, "Safety", "Statically-Proven", origin, "bounded_enabled_access_span", domain, binding, explanation))
        semantic_reason = (
            "shape/stride/storage_offset/element_bytes 与冻结独立参考的物理映射完全一致；"
            "该条件不把两次数值 smoke 提升为一般数值证明。"
        )
        semantic_origin = Origin(point.source_file, point.line, point.kind, point.pointer_param, semantic_reference, binding, semantic_reason)
        candidates.append(Candidate(_metadata_condition(tensor), None, "Semantics", "Statically-Proven", semantic_origin, "frozen_physical_mapping", domain, binding, semantic_reason))
    return ContractResult("Supported", tuple(candidates), tuple(spans), "全部访问点具有有界 span 与冻结物理映射条件")
