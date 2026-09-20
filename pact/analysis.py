"""把统一 Access IR 与外部语义绑定，保留 PoC 的有限候选规则和可追溯输出。"""

import ast
import re
from dataclasses import asdict, dataclass

from pact.access_ir import AccessPoint, Expr, ParseResult, parse_access_ir
from pact.contract_dsl import Condition, ContractClause, Origin, ValueRef
from pact.semantics import OperatorMeaning, SEMANTICS
from scenarios.external_add import add_kernel
from scenarios.kernels import KERNELS


@dataclass(frozen=True)
class Access:
    kind: str
    tensor: str
    offset: str
    expanded_offset: str
    mask: str
    element_bytes: int
    line: int
    source_file: str
    symbols: tuple[str, ...]
    index_calls: tuple[str, ...]
    constexpr: tuple[str, ...]


@dataclass(frozen=True)
class Predicate:
    dimension: int
    expected: str
    category: str
    source: str
    source_access_line: int
    source_access_file: str
    semantic_input: str
    semantic_index: str
    unless_size_one: int | None = None
    tensor: str = "X"


@dataclass(frozen=True)
class Analysis:
    name: str
    status: str
    reason: str
    semantics: OperatorMeaning | None
    accesses: tuple[Access, ...]
    predicates: tuple[Predicate, ...]
    mask_excludes_tail: bool
    refinement_state: str = "尚未发生动态候选修改"
    clauses: tuple[ContractClause, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


def _reject(name: str, meaning: OperatorMeaning | None, status: str, reason: str, points: tuple[AccessPoint, ...] = ()) -> Analysis:
    return Analysis(name, status, reason, meaning, tuple(_legacy(item) for item in points), (), False)


def _legacy(point: AccessPoint) -> Access:
    symbols = tuple(sorted(set(re.findall(r"\b[A-Za-z_]\w*\b", point.expanded_offset)) - {"tl", "program_id", "arange"}))
    return Access(point.kind, point.pointer_param, point.raw_offset, point.expanded_offset, point.expanded_mask, point.element_bytes or 0, point.line, point.source_file, symbols, point.index_calls, point.constexpr)


def _expr(op: str, *items: Expr) -> Expr:
    return Expr(op, args=tuple(sorted(items, key=Expr.key)))


def _param(ir: ParseResult, name: str) -> Expr:
    return Expr("param", ir.signature.index(name))


def _lane(ir: ParseResult, block: str) -> Expr:
    return Expr("lane", args=(_param(ir, block),))


def _linear(ir: ParseResult, block: str) -> Expr:
    return _expr("add", _expr("mul", Expr("pid", 0), _param(ir, block)), _lane(ir, block))


def _row(ir: ParseResult, stride: str, block: str) -> Expr:
    return _expr("add", _expr("mul", Expr("pid", 0), _param(ir, stride)), _lane(ir, block))


def _logical_value(node: ast.AST, meaning: OperatorMeaning) -> Expr | None:
    for item in meaning.inputs:
        try:
            read = ast.parse(item.logical_read, mode="eval").body
        except SyntaxError:
            return None
        if ast.dump(node, include_attributes=False) == ast.dump(read, include_attributes=False):
            return Expr("read", item.tensor)
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return Expr("const", node.value)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _logical_value(node.left, meaning), _logical_value(node.right, meaning)
        return _expr("add", left, right) if left is not None and right is not None else None
    return None


def _binding_check(ir: ParseResult, meaning: OperatorMeaning) -> str | None:
    if not meaning.index_domain or not meaning.logical_write or not meaning.reference_symbol or not meaning.inputs:
        return "缺少可信的独立语义来源"
    pointers = {item.pointer for item in meaning.inputs} | {meaning.output_pointer}
    if len(pointers) != len(meaning.inputs) + 1 or not pointers.issubset(ir.signature):
        return "输入或输出指针绑定不在 Kernel 签名中"
    bindings = dict(meaning.launch_bindings)
    if any(name not in bindings or not bindings[name] for name in ir.signature if name not in pointers):
        return "标量参数缺少独立调用绑定"
    if any(item.tensor != "X" and item.tensor != "Y" for item in meaning.inputs):
        return "旧 Guard 尚不支持这个输入身份"
    if len(ir.accesses) != len(meaning.inputs) + 1 or sum(point.kind == "store" for point in ir.accesses) != 1:
        return "访存点数与独立语义不匹配"
    actual_reads = [point.tensor for point in ir.accesses if point.kind == "load"]
    if sorted(actual_reads) != sorted(item.tensor for item in meaning.inputs):
        return "读取指针与独立语义不匹配"
    if next(point for point in ir.accesses if point.kind == "store").tensor != "OUT":
        return "输出指针与独立语义不匹配"
    if len({point.mask.key() for point in ir.accesses}) != 1:
        return "各访存点的 mask 不一致"
    try:
        expected = _logical_value(ast.parse(meaning.logical_write.split("=", 1)[1].strip(), mode="eval").body, meaning)
    except (IndexError, SyntaxError):
        expected = None
    store = next(point for point in ir.accesses if point.kind == "store")
    if expected is None or store.value is None or expected != store.value:
        return "Kernel 输出计算无法与独立参考语义对应"
    return None


def _shape_binding(ir: ParseResult, meaning: OperatorMeaning) -> str | None:
    """核对逻辑维度到标量实参；D 的元素总数允许两输入共同约束。"""
    bindings = dict(meaning.launch_bindings)
    for item in meaning.inputs:
        try:
            source = ast.parse(item.logical_read, mode="eval").body.value.id
        except (SyntaxError, AttributeError):
            return "逻辑读取不是受支持的 Tensor 索引"
        for axis, param in enumerate(item.shape_params):
            bound = bindings.get(param, "")
            if bound == f"{source}.size({axis})":
                continue
            if len(item.shape_params) == 1 and f"{source}.numel()" in bound:
                continue
            return f"形状参数 {param} 与独立语义的轴绑定冲突"
    return None


def _predicate(meaning: OperatorMeaning, points: tuple[AccessPoint, ...], tensor: str, axis: int, expected: str, explanation: str) -> Predicate:
    item = meaning.input_for(tensor)
    point = next(access for access in points if access.kind == "load" and access.tensor == tensor)
    return Predicate(axis, expected, "语义", explanation, point.line, point.source_file, item.pointer, item.logical_read, tensor=tensor)


def clauses_for(predicates: tuple[Predicate, ...], meaning: OperatorMeaning) -> tuple[ContractClause, ...]:
    """把已推得的旧 PoC 条件编码成 DSL；不会从未证明的类型生成新条件。"""
    clauses = []
    for item in predicates:
        left = ValueRef("stride", tensor=item.tensor, axis=item.dimension)
        right = ValueRef("constant", value=1) if item.expected == "1" else ValueRef("size", tensor=item.tensor, axis=1)
        condition = Condition("eq", left, right)
        if item.unless_size_one is not None:
            singleton = Condition("eq", ValueRef("size", tensor=item.tensor, axis=item.unless_size_one), ValueRef("constant", value=1))
            condition = Condition("or", parts=(singleton, condition))
        origin = Origin(item.source_access_file, item.source_access_line, "load", item.semantic_input, item.semantic_index, meaning.launch_bindings, item.source)
        clauses.append(ContractClause(condition, "Semantics", "Statically-Proven", meaning.assumptions, origin))
    return tuple(clauses)


def extract(name: str) -> Analysis:
    """PoC 兼容入口：统一解析访存，再按结构核对有限语义和候选条件。"""
    meaning = SEMANTICS.get(name)
    # 入口注册表仅负责选源码；不参与地址模板或谓词选择。
    kernel = add_kernel if name == "D" else KERNELS.get(name)
    if kernel is None:
        return _reject(name, meaning, "Unsupported", "未知 Kernel 源码入口")
    if meaning is None:
        return _reject(name, None, "Unknown", "没有可信的独立算子语义来源")
    pointers = {item.pointer: item.tensor for item in meaning.inputs}
    pointers[meaning.output_pointer] = "OUT"
    width = {pointer: 4 for pointer in pointers} if "CUDA float32" in " ".join(meaning.assumptions) else {}
    ir = parse_access_ir(kernel, pointers, width)
    if ir.status != "Supported":
        return _reject(name, meaning, ir.status, ir.reason, ir.accesses)
    reason = _binding_check(ir, meaning) or _shape_binding(ir, meaning)
    if reason:
        return _reject(name, meaning, "Unknown", reason, ir.accesses)
    points = ir.accesses
    store = next(point for point in points if point.kind == "store")
    loads = tuple(point for point in points if point.kind == "load")
    bindings = dict(meaning.launch_bindings)
    shape = meaning.inputs[0].shape_params
    if not all(param in ir.signature for param in shape):
        return _reject(name, meaning, "Unknown", "逻辑形状参数不在 Kernel 签名中", points)
    bound = _param(ir, shape[0]) if len(shape) == 1 else _expr("mul", *(_param(ir, param) for param in shape))
    mask = Expr("lt", args=(store.offset, bound))
    predicates: tuple[Predicate, ...]
    if all(point.offset == store.offset == _linear(ir, ir.signature[-1]) for point in points) and all(point.mask == mask for point in points):
        if meaning.ndim == 2 and len(loads) == 1:
            predicates = (
                _predicate(meaning, points, loads[0].tensor, 0, "size(1)", "线性地址与二维行优先逻辑索引对应"),
                _predicate(meaning, points, loads[0].tensor, 1, "1", "相邻列的读取地址差为 1"),
            )
        elif meaning.ndim == 1:
            predicates = tuple(_predicate(meaning, points, point.tensor, 0, "1", "线性地址与逐元素逻辑索引对应") for point in loads)
        else:
            return _reject(name, meaning, "Unsupported", "线性访问与语义维度不匹配", points)
    elif meaning.ndim == 2 and len(loads) == 1:
        # 行步长版本：行地址来自输入 stride(0)，输出行地址来自逻辑列数。
        block = ir.signature[-1]
        row_input = next((param for param in ir.signature if param not in {item.pointer for item in meaning.inputs} | {meaning.output_pointer} | set(shape) | {block}), None)
        source = ast.parse(meaning.inputs[0].logical_read, mode="eval").body.value.id
        if row_input is None or bindings.get(row_input) != f"{source}.stride(0)" or loads[0].offset != _row(ir, row_input, block) or store.offset != _row(ir, shape[1], block):
            return _reject(name, meaning, "Unknown", "行步长地址或 stride 实参绑定不匹配", points)
        row_mask = Expr("lt", args=(_lane(ir, block), _param(ir, shape[1])))
        if any(point.mask != row_mask for point in points):
            return _reject(name, meaning, "Unsupported", "行 Tile 的 mask 不能证明尾部覆盖", points)
        predicates = (_predicate(meaning, points, loads[0].tensor, 1, "1", "行步长来自输入，列地址以单位步长递增"),)
    else:
        return _reject(name, meaning, "Unsupported", "访存地址不属于 PoC 候选规则的支持域", points)
    return Analysis(name, "Supported", "统一 Access IR 与独立语义、参数绑定及 PoC 候选规则匹配", meaning, tuple(_legacy(item) for item in points), predicates, True, clauses=clauses_for(predicates, meaning))
