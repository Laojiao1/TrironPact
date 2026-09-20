"""保守解释 tl.multiple_of 的实际操作数；结果仅供离线和影子核对。"""

from __future__ import annotations

from dataclasses import dataclass

from pact.access_ir import AlignmentHint, Expr, ParseResult
from pact.analysis import _binding_check, _shape_binding
from pact.candidates import BindingResult, CallBinding, Candidate, validate_binding
from pact.contract_dsl import Condition, Origin, ValueRef
from pact.semantics import OperatorMeaning


@dataclass(frozen=True)
class AlignmentResult:
    status: str
    candidates: tuple[Candidate, ...]
    reason: str


def _mul(left: Expr, right: Expr) -> Expr:
    return Expr("mul", args=tuple(sorted((left, right), key=Expr.key)))


def _origin(point, meaning: OperatorMeaning, reason: str) -> Origin:
    semantic = meaning.logical_write if point.kind == "store" else meaning.input_for(point.tensor).logical_read
    return Origin(point.source_file, point.line, point.kind, point.pointer_param, semantic, meaning.launch_bindings, reason)


def _unknown(hint: AlignmentHint, point, meaning: OperatorMeaning, reason: str) -> Candidate:
    return Candidate(None, hint.input, "Optimization", "Unknown", _origin(point, meaning, reason), "multiple_of_operand", meaning.assumptions or ("未确认支持域",), meaning.launch_bindings, reason)


def extract_alignment(ir: ParseResult, meaning: OperatorMeaning | None, call: CallBinding, call_site: BindingResult) -> AlignmentResult:
    if not ir.hints:
        return AlignmentResult("Supported", (), "没有 tl.multiple_of 提示")
    if meaning is None:
        return AlignmentResult("Unknown", (), "缺少独立算子语义")
    checked = validate_binding(ir, meaning, call)
    if checked.status != "Supported" or call_site.status != "Supported":
        return AlignmentResult("Unknown", (), "; ".join(checked.reasons + call_site.reasons) or "调用绑定未核对")
    semantic_reason = _binding_check(ir, meaning) or _shape_binding(ir, meaning)
    if semantic_reason:
        return AlignmentResult("Unknown", (), semantic_reason)
    records = []
    for hint in ir.hints:
        points = [point for point in ir.accesses if point.line in hint.used_access_lines]
        if not points:
            # 没有关联访问时只保留 IR 提示，不声称任何输入契约。
            return AlignmentResult("Unknown", tuple(records), f"{hint.source_file}:{hint.line} 的提示未作用于访存地址或 mask")
        point = points[0]
        source = f"{hint.source_file}:{hint.line} 的 tl.multiple_of({hint.raw_input}, {hint.multiple})"
        if hint.pointer_param is not None:
            point = next((item for item in points if item.pointer_param == hint.pointer_param), None)
            if point is None or point.element_bytes is None or point.element_bytes < 1:
                records.append(_unknown(hint, points[0], meaning, source + " 未匹配已知宽度的指针访存"))
                continue
            condition = Condition("mod_eq", ValueRef("effective_ptr", tensor=point.tensor), divisor=hint.multiple, remainder=0)
            reason = source + " 作用于有效基址；指针模数按字节计，data_ptr 已含视图偏移，不再叠加 storage_offset；动态地址尚未静态保证"
            records.append(Candidate(condition, None, "Optimization", "Unproven", _origin(point, meaning, reason), "multiple_of_pointer_base", meaning.assumptions, meaning.launch_bindings, reason))
            continue
        scalar = next((name for name, _ in call.scalars if name in ir.signature and hint.input == _mul(Expr("pid", 0), Expr("param", ir.signature.index(name)))), None)
        if scalar is None:
            records.append(_unknown(hint, point, meaning, source + " 的组首不是受支持的标量 pid(0)*Tile 形式"))
            continue
        bound = dict(call.scalars)[scalar]
        condition = Condition("mod_eq", ValueRef("scalar", scalar=scalar), divisor=hint.multiple, remainder=0)
        proven = bound.isdigit() and int(bound) > 0 and int(bound) % hint.multiple == 0
        reason = source + f" 仅约束索引组首 pid(0)*{scalar} 的元素偏移；不推出任何 Tensor 的有效指针对齐"
        records.append(Candidate(condition, None, "Optimization", "Statically-Proven" if proven else "Unproven", _origin(point, meaning, reason), "multiple_of_index_start", meaning.assumptions, meaning.launch_bindings, reason))
    return AlignmentResult("Unknown" if any(item.evidence == "Unknown" for item in records) else "Supported", tuple(records), "提示作用对象已分类；不授予 Fast 资格")
