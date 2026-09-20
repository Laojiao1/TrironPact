"""把受限 Triton Python AST 归一化为与案例名称无关的访存 Access IR。"""

from __future__ import annotations

import ast
import copy
import inspect
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path


class ParseFailure(Exception):
    def __init__(self, status: str, reason: str):
        super().__init__(reason)
        self.status = status
        self.reason = reason


@dataclass(frozen=True)
class Expr:
    op: str
    value: int | str | None = None
    args: tuple[Expr, ...] = ()

    def key(self) -> str:
        return f"{self.op}({self.value if self.value is not None else ','.join(item.key() for item in self.args)})"

    def contains_index(self) -> bool:
        return self.op in ("pid", "lane") or any(item.contains_index() for item in self.args)


def _commutative(op: str, items: tuple[Expr, ...]) -> Expr:
    flat = tuple(part for item in items for part in (item.args if item.op == op else (item,)))
    return Expr(op, args=tuple(sorted(flat, key=Expr.key)))


@dataclass(frozen=True)
class AccessPoint:
    kind: str
    pointer_param: str
    tensor: str
    raw_address: str
    raw_offset: str
    expanded_offset: str
    offset: Expr
    raw_mask: str
    expanded_mask: str
    mask: Expr
    source_file: str
    line: int
    element_bytes: int | None
    index_calls: tuple[str, ...]
    constexpr: tuple[str, ...]
    value: Expr | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AlignmentHint:
    raw_input: str
    input: Expr
    multiple: int
    assigned_name: str
    source_file: str
    line: int
    pointer_param: str | None
    used_access_lines: tuple[int, ...]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ParseResult:
    status: str
    reason: str
    signature: tuple[str, ...]
    accesses: tuple[AccessPoint, ...]
    store_value_supported: bool = False
    hints: tuple[AlignmentHint, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


def _tl_call(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "tl" and node.func.attr == name


def _source(fn_or_source) -> tuple[str, int, str]:
    if isinstance(fn_or_source, str):
        return textwrap.dedent(fn_or_source), 1, "<provided>"
    fn = getattr(fn_or_source, "fn", fn_or_source)
    lines, first = inspect.getsourcelines(fn)
    file = Path(inspect.getsourcefile(fn)).resolve()
    root = Path(__file__).resolve().parent.parent
    try:
        name = file.relative_to(root).as_posix()
    except ValueError:
        name = file.as_posix()
    return textwrap.dedent("".join(lines)), first, name


class _Normalizer:
    def __init__(self, signature: tuple[str, ...], assignments: dict[str, ast.AST], constexpr: tuple[str, ...]):
        self.params = {name: pos for pos, name in enumerate(signature)}
        self.assignments = assignments
        self.constexpr = set(constexpr)

    def expr(self, node: ast.AST, seen: frozenset[str] = frozenset()) -> Expr:
        if _tl_call(node, "multiple_of") and len(node.args) == 2 and not node.keywords:
            return self.expr(node.args[0], seen)
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return Expr("const", node.value)
        if isinstance(node, ast.Name):
            if node.id in self.assignments:
                if node.id in seen:
                    raise ParseFailure("Unsupported", "索引变量存在循环定义")
                return self.expr(self.assignments[node.id], seen | {node.id})
            if node.id in self.params:
                return Expr("param", self.params[node.id])
            raise ParseFailure("Unknown", f"未绑定的索引变量：{node.id}")
        if _tl_call(node, "program_id"):
            axis = node.args[0] if len(node.args) == 1 and not node.keywords else (node.keywords[0].value if not node.args and len(node.keywords) == 1 and node.keywords[0].arg == "axis" else None)
            if not isinstance(axis, ast.Constant) or type(axis.value) is not int or axis.value != 0:
                raise ParseFailure("Unsupported", "只支持 program_id 的第 0 轴")
            return Expr("pid", axis.value)
        if _tl_call(node, "arange"):
            if len(node.args) != 2 or not isinstance(node.args[0], ast.Constant) or node.args[0].value != 0:
                raise ParseFailure("Unsupported", "只支持从 0 开始的 arange")
            if isinstance(node.args[1], ast.Name) and node.args[1].id not in self.constexpr:
                raise ParseFailure("Unknown", "arange 块大小缺少 constexpr 声明")
            block = self.expr(node.args[1], seen)
            if block.op not in ("param", "const"):
                raise ParseFailure("Unsupported", "arange 的块大小需要静态标量")
            return Expr("lane", args=(block,))
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mult)):
            left, right = self.expr(node.left, seen), self.expr(node.right, seen)
            if isinstance(node.op, ast.Add):
                return _commutative("add", (left, right))
            if left.contains_index() and right.contains_index():
                raise ParseFailure("Unsupported", "两个索引量相乘不在仿射支持域")
            return _commutative("mul", (left, right))
        raise ParseFailure("Unsupported", f"不支持的索引表达式：{ast.unparse(node)}")

    def mask(self, node: ast.AST) -> Expr:
        if isinstance(node, ast.Name) and node.id in self.assignments:
            return self.mask(self.assignments[node.id])
        if isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1 and isinstance(node.ops[0], ast.Lt):
            left, right = self.expr(node.left), self.expr(node.comparators[0])
            if not left.contains_index() or right.contains_index():
                raise ParseFailure("Unsupported", "mask 不是受支持的尾部边界比较")
            return Expr("lt", args=(left, right))
        raise ParseFailure("Unsupported", "mask 不是受支持的显式比较")

    def value(self, node: ast.AST, load_names: dict[str, str], seen: frozenset[str] = frozenset()) -> Expr | None:
        if isinstance(node, ast.Name):
            if node.id in load_names:
                return Expr("read", load_names[node.id])
            if node.id in self.assignments and node.id not in seen:
                return self.value(self.assignments[node.id], load_names, seen | {node.id})
            return None
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return Expr("const", node.value)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            left, right = self.value(node.left, load_names, seen), self.value(node.right, load_names, seen)
            return _commutative("add", (left, right)) if left is not None and right is not None else None
        return None


def _add_parts(node: ast.AST) -> list[ast.AST]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _add_parts(node.left) + _add_parts(node.right)
    return [node]


def _expanded(node: ast.AST, assignments: dict[str, ast.AST], seen: frozenset[str] = frozenset()) -> ast.AST:
    """仅用于报告源码形式；规范化判断始终使用 Expr。"""
    if isinstance(node, ast.Name) and node.id in assignments and node.id not in seen:
        return _expanded(assignments[node.id], assignments, seen | {node.id})
    result = copy.deepcopy(node)
    for field, value in ast.iter_fields(result):
        if isinstance(value, ast.AST):
            setattr(result, field, _expanded(value, assignments, seen))
        elif isinstance(value, list):
            setattr(result, field, [_expanded(item, assignments, seen) if isinstance(item, ast.AST) else item for item in value])
    return result


def _pointer_base(node: ast.AST, assignments: dict[str, ast.AST], pointers: dict[str, str]) -> str | None:
    if not isinstance(node, ast.Name):
        return None
    if node.id in pointers:
        return node.id
    assigned = assignments.get(node.id)
    if _tl_call(assigned, "multiple_of") and len(assigned.args) == 2 and isinstance(assigned.args[0], ast.Name) and assigned.args[0].id in pointers:
        return assigned.args[0].id
    return None


def _used_names(node: ast.AST, assignments: dict[str, ast.AST], seen: frozenset[str] = frozenset()) -> set[str]:
    names = {item.id for item in ast.walk(node) if isinstance(item, ast.Name)}
    result = set(names)
    for name in names & assignments.keys() - seen:
        result.update(_used_names(assignments[name], assignments, seen | {name}))
    return result


def parse_access_ir(fn_or_source, pointer_bindings: dict[str, str], element_bytes: dict[str, int] | None = None) -> ParseResult:
    """只确认受支持的访存语法；语义绑定和 Fast 资格由上层另行核对。"""
    try:
        source, first, file = _source(fn_or_source)
    except (TypeError, OSError, ValueError):
        return ParseResult("Unknown", "无法读取可信的 Triton 源码", (), ())
    try:
        tree = ast.parse(source)
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    except (SyntaxError, StopIteration):
        return ParseResult("Unsupported", "没有可解析的 Triton 函数", (), ())
    signature = tuple(arg.arg for arg in function.args.args)
    if not pointer_bindings or any(name not in signature for name in pointer_bindings) or len(set(pointer_bindings.values())) != len(pointer_bindings):
        return ParseResult("Unknown", "指针绑定缺失、重复或不在函数签名中", signature, ())
    assignments: dict[str, ast.AST] = {}
    calls: list[tuple[str, ast.Call]] = []
    hint_nodes: list[tuple[str, ast.Call]] = []
    body = function.body[1:] if function.body and isinstance(function.body[0], ast.Expr) and isinstance(function.body[0].value, ast.Constant) and isinstance(function.body[0].value.value, str) else function.body
    for statement in body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
            target = statement.targets[0].id
            if target in assignments or target in signature:
                return ParseResult("Unsupported", f"第 {first + statement.lineno - 1} 行重复定义变量或覆盖参数", signature, ())
            assignments[target] = statement.value
            if _tl_call(statement.value, "multiple_of"):
                hint_nodes.append((target, statement.value))
            if _tl_call(statement.value, "load"):
                calls.append(("load", statement.value))
        elif isinstance(statement, ast.Expr) and _tl_call(statement.value, "store"):
            calls.append(("store", statement.value))
        else:
            return ParseResult("Unsupported", f"不支持第 {first + statement.lineno - 1} 行的控制流或语句", signature, ())
    if sum(_tl_call(node, "multiple_of") for node in ast.walk(function)) != len(hint_nodes):
        return ParseResult("Unsupported", "tl.multiple_of 必须作为独立赋值，不能嵌套使用", signature, ())
    if not calls or not any(kind == "load" for kind, _ in calls) or not any(kind == "store" for kind, _ in calls):
        return ParseResult("Unsupported", "缺少显式 load/store 访问", signature, ())
    for op in ("load", "store"):
        if sum(_tl_call(node, op) for node in ast.walk(function)) != sum(kind == op for kind, _ in calls):
            return ParseResult("Unsupported", "访存调用存在未建模的嵌套形式", signature, ())
    constexpr = tuple(arg.arg for arg in function.args.args if isinstance(arg.annotation, ast.Attribute) and isinstance(arg.annotation.value, ast.Name) and arg.annotation.value.id == "tl" and arg.annotation.attr == "constexpr")
    normalizer = _Normalizer(signature, assignments, constexpr)
    accesses: list[AccessPoint] = []
    load_names: dict[str, str] = {}
    try:
        hints: list[AlignmentHint] = []
        for name, hint in hint_nodes:
            if len(hint.args) != 2 or hint.keywords or not isinstance(hint.args[1], ast.Constant) or type(hint.args[1].value) is not int or hint.args[1].value < 1 or hint.args[1].value & (hint.args[1].value - 1):
                raise ParseFailure("Unsupported", "tl.multiple_of 只支持字面量正二次幂倍数")
            target = normalizer.expr(hint.args[0])
            pointer = hint.args[0].id if isinstance(hint.args[0], ast.Name) and hint.args[0].id in pointer_bindings else None
            hints.append(AlignmentHint(ast.unparse(hint.args[0]), target, hint.args[1].value, name, file, first + hint.lineno - 1, pointer, ()))
        for kind, call in calls:
            if not call.args:
                raise ParseFailure("Unsupported", "访存调用缺少指针")
            pointer_parts = _add_parts(call.args[0])
            bases = [(_pointer_base(item, assignments, pointer_bindings), item) for item in pointer_parts]
            bases = [(base, item) for base, item in bases if base is not None]
            if len(bases) != 1:
                raise ParseFailure("Unknown", "访存基址缺少唯一指针绑定")
            base, base_node = bases[0]
            remaining = list(pointer_parts)
            remaining.remove(base_node)
            if not remaining:
                raise ParseFailure("Unsupported", "访存地址缺少索引")
            offset = _commutative("add", tuple(normalizer.expr(item) for item in remaining)) if len(remaining) > 1 else normalizer.expr(remaining[0])
            mask_node = next((item.value for item in call.keywords if item.arg == "mask"), None)
            if mask_node is None:
                raise ParseFailure("Unsupported", "tl.load/store 缺少显式 mask")
            mask = normalizer.mask(mask_node)
            width = (element_bytes or {}).get(base)
            if width is not None and (type(width) is not int or width < 1):
                raise ParseFailure("Unknown", "元素宽度绑定无效")
            index_calls = tuple(name for name, op in (("program_id", "pid"), ("arange", "lane")) if op in offset.key())
            raw_offset = " + ".join(ast.unparse(item) for item in remaining)
            expanded_offset = " + ".join(ast.unparse(_expanded(item, assignments)) for item in remaining)
            point = AccessPoint(kind, base, pointer_bindings[base], ast.unparse(call.args[0]), raw_offset, expanded_offset, offset, ast.unparse(mask_node), ast.unparse(_expanded(mask_node, assignments)), mask, file, first + call.lineno - 1, width, index_calls, constexpr)
            accesses.append(point)
            if kind == "load":
                names = [name for name, value in assignments.items() if value is call]
                if len(names) != 1:
                    raise ParseFailure("Unknown", "load 结果没有唯一变量")
                load_names[names[0]] = pointer_bindings[base]
        stores = [point for point in accesses if point.kind == "store"]
        if len(stores) != 1:
            raise ParseFailure("Unsupported", "当前只支持一个 store 访问点")
        store_call = next(call for kind, call in calls if kind == "store")
        value = normalizer.value(store_call.args[1], load_names) if len(store_call.args) > 1 else None
        index = accesses.index(stores[0])
        point = accesses[index]
        accesses[index] = AccessPoint(point.kind, point.pointer_param, point.tensor, point.raw_address, point.raw_offset, point.expanded_offset, point.offset, point.raw_mask, point.expanded_mask, point.mask, point.source_file, point.line, point.element_bytes, point.index_calls, point.constexpr, value)
        associated = []
        for hint in hints:
            lines = tuple(first + call.lineno - 1 for _, call in calls if hint.assigned_name in _used_names(call.args[0], assignments) or any(hint.assigned_name in _used_names(keyword.value, assignments) for keyword in call.keywords if keyword.arg == "mask"))
            associated.append(AlignmentHint(hint.raw_input, hint.input, hint.multiple, hint.assigned_name, hint.source_file, hint.line, hint.pointer_param, lines))
        return ParseResult("Supported", "访存语法已归一化；尚未核对独立算子语义", signature, tuple(accesses), value is not None, tuple(associated))
    except ParseFailure as error:
        return ParseResult(error.status, error.reason, signature, tuple(accesses))
