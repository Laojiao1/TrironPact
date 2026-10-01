"""e2 的绑定后 Access IR 提取器。

该模块只恢复访存地址、mask 和访问点归属，不解释 Kernel 的数值计算。
调用方必须提供指针绑定以及会改变访存路径的 constexpr 值；缺失绑定、
动态控制流、间接索引或无法证明的地址形式均返回 ``Unknown/Unsupported``。
"""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass

from pact.access_ir import AccessPoint, Expr, ParseFailure, ParseResult, _commutative, _source, _tl_call


@dataclass(frozen=True)
class _Pointer:
    """指针参数及其元素偏移；offset 的单位始终是元素而不是字节。"""

    base: str
    offset: Expr


def _const(value: int) -> Expr:
    return Expr("const", value)


def _add(left: Expr, right: Expr) -> Expr:
    if left.op == right.op == "const":
        return _const(int(left.value) + int(right.value))
    if left == _const(0):
        return right
    if right == _const(0):
        return left
    return _commutative("add", (left, right))


class _BoundParser:
    def __init__(
        self,
        signature: tuple[str, ...],
        pointers: dict[str, str],
        constexpr: set[str],
        constexpr_values: dict[str, int | bool],
        element_bytes: dict[str, int],
        source_file: str,
        first_line: int,
    ) -> None:
        self.signature = signature
        self.params = {name: index for index, name in enumerate(signature)}
        self.pointer_bindings = pointers
        self.constexpr = constexpr
        self.constexpr_values = constexpr_values
        self.element_bytes = element_bytes
        self.source_file = source_file
        self.first_line = first_line
        self.values: dict[str, ast.AST] = {}
        self.pointer_values: dict[str, _Pointer] = {
            name: _Pointer(name, _const(0)) for name in pointers
        }
        self.accesses: list[AccessPoint] = []
        self.load_names: dict[int, str] = {}

    def _line(self, node: ast.AST) -> int:
        return self.first_line + getattr(node, "lineno", 1) - 1

    def _expanded(self, node: ast.AST, seen: frozenset[str] = frozenset()) -> ast.AST:
        if isinstance(node, ast.Name) and node.id in self.values and node.id not in seen:
            return self._expanded(self.values[node.id], seen | {node.id})
        result = copy.deepcopy(node)
        for field, value in ast.iter_fields(result):
            if isinstance(value, ast.AST):
                setattr(result, field, self._expanded(value, seen))
            elif isinstance(value, list):
                setattr(result, field, [self._expanded(item, seen) if isinstance(item, ast.AST) else item for item in value])
        return result

    def _constexpr_value(self, node: ast.AST) -> int | bool | None:
        if isinstance(node, ast.Constant) and type(node.value) in (int, bool):
            return node.value
        if isinstance(node, ast.Name):
            if node.id in self.constexpr_values:
                return self.constexpr_values[node.id]
            if node.id in self.values:
                return self._constexpr_value(self.values[node.id])
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            value = self._constexpr_value(node.operand)
            return None if value is None else not bool(value)
        if isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1:
            left = self._constexpr_value(node.left)
            right = self._constexpr_value(node.comparators[0])
            if left is None or right is None:
                return None
            if isinstance(node.ops[0], ast.Eq):
                return left == right
            if isinstance(node.ops[0], ast.NotEq):
                return left != right
        return None

    def expr(self, node: ast.AST, seen: frozenset[str] = frozenset()) -> Expr:
        if isinstance(node, ast.Constant) and type(node.value) in (int, bool):
            return _const(int(node.value))
        if isinstance(node, ast.Name):
            if node.id in self.values:
                if node.id in seen:
                    raise ParseFailure("Unsupported", "索引变量存在循环定义")
                return self.expr(self.values[node.id], seen | {node.id})
            if node.id in self.constexpr_values:
                return _const(int(self.constexpr_values[node.id]))
            if node.id in self.params and node.id not in self.pointer_bindings:
                return Expr("param", self.params[node.id])
            raise ParseFailure("Unknown", f"未绑定的索引变量：{node.id}")
        if _tl_call(node, "program_id"):
            axis_node = node.args[0] if len(node.args) == 1 else next(
                (item.value for item in node.keywords if item.arg == "axis"), None
            )
            if not isinstance(axis_node, ast.Constant) or type(axis_node.value) is not int or axis_node.value not in (0, 1):
                raise ParseFailure("Unsupported", "只支持 program_id 的第 0/1 轴")
            return Expr("pid", axis_node.value)
        if _tl_call(node, "arange"):
            if len(node.args) != 2 or not isinstance(node.args[0], ast.Constant) or node.args[0].value != 0:
                raise ParseFailure("Unsupported", "只支持从 0 开始的 arange")
            return Expr("lane", args=(self.expr(node.args[1], seen),))
        if isinstance(node, ast.Subscript):
            slices = node.slice.elts if isinstance(node.slice, ast.Tuple) else (node.slice,)
            if not all(
                isinstance(item, ast.Constant) and item.value is None
                or isinstance(item, ast.Slice) and item.lower is None and item.upper is None and item.step is None
                for item in slices
            ):
                raise ParseFailure("Unsupported", "只支持 None 与完整切片构成的广播下标")
            base = self.expr(node.value, seen)
            # None 维只改变张量广播形状，不改变扁平元素坐标。
            dimensions = ast.unparse(node.slice)
            return Expr("broadcast", dimensions, (base,))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return _commutative("mul", (_const(-1), self.expr(node.operand, seen)))
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv)):
            left, right = self.expr(node.left, seen), self.expr(node.right, seen)
            if isinstance(node.op, ast.Add):
                return _add(left, right)
            if isinstance(node.op, ast.Sub):
                if left.op == right.op == "const":
                    return _const(int(left.value) - int(right.value))
                return _add(left, _commutative("mul", (_const(-1), right)))
            if isinstance(node.op, ast.Mult):
                if left.op == right.op == "const":
                    return _const(int(left.value) * int(right.value))
                if left.contains_index() and right.contains_index():
                    raise ParseFailure("Unsupported", "两个索引量相乘不在仿射支持域")
                return _commutative("mul", (left, right))
            if right.op != "const" or not isinstance(right.value, int) or right.value <= 0:
                raise ParseFailure("Unsupported", "索引整除仅支持正整数字面量除数")
            if left.op == "const":
                return _const(int(left.value) // int(right.value))
            return Expr("floordiv", right.value, (left,))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in ("to", "cast"):
            target = node.args[0] if len(node.args) == 1 else None
            integer_target = (
                isinstance(target, ast.Attribute)
                and isinstance(target.value, ast.Name)
                and target.value.id == "tl"
                and target.attr in ("int32", "int64", "uint32", "uint64")
            )
            if integer_target:
                return self.expr(node.func.value, seen)
            raise ParseFailure("Unsupported", "地址 cast 只支持显式 Triton 整数类型")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "reshape":
            if node.args and isinstance(node.func.value, ast.AST):
                return self.expr(node.func.value, seen)
        if (_tl_call(node, "reshape") or _tl_call(node, "ravel")) and node.args:
            # reshape/ravel 只改变 block tensor 的视图形状；这里恢复的是扁平元素坐标。
            return self.expr(node.args[0], seen)
        if _tl_call(node, "where") and len(node.args) == 3:
            selected = self._constexpr_value(node.args[0])
            if selected is None:
                raise ParseFailure("Unknown", "地址 tl.where 条件不是已绑定 constexpr")
            return self.expr(node.args[1] if selected else node.args[2], seen)
        raise ParseFailure("Unsupported", f"不支持的索引表达式：{ast.unparse(node)}")

    def mask(self, node: ast.AST | None) -> Expr:
        if node is None:
            return Expr("true")
        if isinstance(node, ast.Name) and node.id in self.values:
            return self.mask(self.values[node.id])
        if isinstance(node, ast.Constant) and type(node.value) is bool:
            return Expr("true" if node.value else "false")
        if isinstance(node, ast.Subscript):
            slices = node.slice.elts if isinstance(node.slice, ast.Tuple) else (node.slice,)
            if not all(
                isinstance(item, ast.Constant) and item.value is None
                or isinstance(item, ast.Slice) and item.lower is None and item.upper is None and item.step is None
                for item in slices
            ):
                raise ParseFailure("Unsupported", "mask 只支持 None 与完整切片广播")
            return Expr("broadcast", ast.unparse(node.slice), (self.mask(node.value),))
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitAnd):
            return _commutative("and", (self.mask(node.left), self.mask(node.right)))
        if isinstance(node, ast.BoolOp) and isinstance(node.op, ast.And):
            return _commutative("and", tuple(self.mask(item) for item in node.values))
        if isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1 and isinstance(node.ops[0], ast.Lt):
            left, right = self.expr(node.left), self.expr(node.comparators[0])
            if not left.contains_index() or right.contains_index():
                raise ParseFailure("Unsupported", "mask 不是受支持的行列边界比较")
            return Expr("lt", args=(left, right))
        raise ParseFailure("Unsupported", f"mask 不是受支持的基本边界组合：{ast.unparse(node)}")

    def pointer(self, node: ast.AST, seen: frozenset[str] = frozenset()) -> _Pointer:
        if isinstance(node, ast.Name):
            if node.id in self.pointer_values:
                return self.pointer_values[node.id]
            if node.id in self.values:
                if node.id in seen:
                    raise ParseFailure("Unsupported", "指针变量存在循环定义")
                return self.pointer(self.values[node.id], seen | {node.id})
            raise ParseFailure("Unknown", f"访存基址缺少唯一指针绑定：{node.id}")
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            for pointer_node, offset_node, sign in (
                (node.left, node.right, 1),
                (node.right, node.left, 1 if isinstance(node.op, ast.Add) else None),
            ):
                if sign is None:
                    continue
                try:
                    pointer = self.pointer(pointer_node, seen)
                except ParseFailure:
                    continue
                offset = self.expr(offset_node)
                if isinstance(node.op, ast.Sub):
                    offset = _commutative("mul", (_const(-1), offset))
                return _Pointer(pointer.base, _add(pointer.offset, offset))
        raise ParseFailure("Unknown", "访存基址缺少唯一指针绑定")

    def _record(self, kind: str, call: ast.Call, assigned_name: str | None = None) -> None:
        if not call.args:
            raise ParseFailure("Unsupported", "访存调用缺少指针")
        pointer = self.pointer(call.args[0])
        mask_node = next((item.value for item in call.keywords if item.arg == "mask"), None)
        if mask_node is None:
            position = 1 if kind == "load" else 2
            if len(call.args) > position:
                mask_node = call.args[position]
        mask = self.mask(mask_node)
        width = self.element_bytes.get(pointer.base)
        if width is None or type(width) is not int or width < 1:
            raise ParseFailure("Unknown", f"指针 {pointer.base} 缺少有效元素宽度")
        raw_mask = ast.unparse(mask_node) if mask_node is not None else "True"
        expanded_mask = ast.unparse(self._expanded(mask_node)) if mask_node is not None else "True"
        index_calls: list[str] = []
        key = pointer.offset.key()
        for axis in (0, 1):
            if f"pid({axis})" in key:
                index_calls.append(f"program_id({axis})")
        if "lane(" in key:
            index_calls.append("arange")
        point = AccessPoint(
            kind=kind,
            pointer_param=pointer.base,
            tensor=self.pointer_bindings[pointer.base],
            raw_address=ast.unparse(call.args[0]),
            raw_offset=ast.unparse(call.args[0]),
            expanded_offset=ast.unparse(self._expanded(call.args[0])),
            offset=pointer.offset,
            raw_mask=raw_mask,
            expanded_mask=expanded_mask,
            mask=mask,
            source_file=self.source_file,
            line=self._line(call),
            element_bytes=width,
            index_calls=tuple(index_calls),
            constexpr=tuple(sorted(self.constexpr)),
        )
        self.accesses.append(point)
        if kind == "load" and assigned_name:
            self.load_names[id(call)] = assigned_name

    @staticmethod
    def _memory_calls(node: ast.AST) -> list[tuple[str, ast.Call]]:
        calls: list[tuple[str, ast.Call]] = []
        for item in ast.walk(node):
            if isinstance(item, ast.Call):
                if _tl_call(item, "load"):
                    calls.append(("load", item))
                elif _tl_call(item, "store"):
                    calls.append(("store", item))
        return sorted(calls, key=lambda pair: (getattr(pair[1], "lineno", 0), getattr(pair[1], "col_offset", 0)))

    def statement(self, statement: ast.stmt) -> None:
        if isinstance(statement, ast.If):
            selected = self._constexpr_value(statement.test)
            if selected is None:
                # 即使分支本身不访存，也可能更新后续地址所依赖的指针或标量。
                raise ParseFailure("Unknown", "控制流条件缺少可证明的 constexpr 绑定")
            branch = statement.body if bool(selected) else statement.orelse
            for child in branch:
                self.statement(child)
            return
        if isinstance(statement, ast.AugAssign) and isinstance(statement.target, ast.Name) and isinstance(statement.op, (ast.Add, ast.Sub)):
            name = statement.target.id
            if name not in self.pointer_values:
                raise ParseFailure("Unsupported", f"只支持指针参数的增量更新：{name}")
            pointer = self.pointer_values[name]
            delta = self.expr(statement.value)
            if isinstance(statement.op, ast.Sub):
                delta = _commutative("mul", (_const(-1), delta))
            self.pointer_values[name] = _Pointer(pointer.base, _add(pointer.offset, delta))
            return
        if isinstance(statement, (ast.Assign, ast.AnnAssign)):
            targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
            value = statement.value
            if value is None:
                raise ParseFailure("Unsupported", "赋值缺少右值")
            if len(targets) != 1 or not isinstance(targets[0], ast.Name):
                if self._memory_calls(value):
                    raise ParseFailure("Unknown", "含访存的解构赋值未建模")
                # 数值张量的解构不会改变访存地址；数值语义由独立参考验证。
                return
            name = targets[0].id
            calls = self._memory_calls(value)
            for kind, call in calls:
                self._record(kind, call, name if kind == "load" and call is value else None)
            try:
                pointer = self.pointer(value)
            except ParseFailure:
                self.values[name] = value
                self.pointer_values.pop(name, None)
            else:
                self.pointer_values[name] = pointer
                self.values.pop(name, None)
            return
        if isinstance(statement, ast.Expr):
            for kind, call in self._memory_calls(statement.value):
                self._record(kind, call)
            return
        if isinstance(statement, (ast.Pass, ast.Return)):
            if self._memory_calls(statement):
                raise ParseFailure("Unsupported", "return 中的访存未建模")
            return
        raise ParseFailure("Unsupported", f"不支持第 {self._line(statement)} 行的动态控制流或语句")


def parse_extended_access_ir(
    fn_or_source,
    pointer_bindings: dict[str, str],
    element_bytes: dict[str, int],
    constexpr_values: dict[str, int | bool] | None = None,
) -> ParseResult:
    """提取调用绑定后的多访问点 IR。

    ``constexpr_values`` 只用于选择编译期分支和折叠地址条件。返回 Supported
    仅说明全部 load/store 的访问域已建模；它不证明 reduction 或普通算术的数值语义，
    也不授予 Guard/Fast 资格。
    """

    try:
        source, first, file = _source(fn_or_source)
        tree = ast.parse(source)
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    except (TypeError, OSError, ValueError):
        return ParseResult("Unknown", "无法读取可信的 Triton 源码", (), ())
    except (SyntaxError, StopIteration):
        return ParseResult("Unsupported", "没有可解析的 Triton 函数", (), ())
    signature = tuple(arg.arg for arg in function.args.args)
    if not pointer_bindings or any(name not in signature for name in pointer_bindings):
        return ParseResult("Unknown", "指针绑定缺失或不在函数签名中", signature, ())
    if len(set(pointer_bindings.values())) != len(pointer_bindings):
        return ParseResult("Unknown", "多个指针绑定到同一逻辑张量；alias 未证明", signature, ())
    constexpr = {
        arg.arg
        for arg in function.args.args
        if (
            isinstance(arg.annotation, ast.Attribute)
            and isinstance(arg.annotation.value, ast.Name)
            and arg.annotation.value.id == "tl"
            and arg.annotation.attr == "constexpr"
        )
        or (
            isinstance(arg.annotation, ast.Constant)
            and arg.annotation.value == "tl.constexpr"
        )
    }
    values = constexpr_values or {}
    if any(name in signature and name not in constexpr for name in values):
        return ParseResult("Unknown", "constexpr 绑定包含非 constexpr 形参", signature, ())
    parser = _BoundParser(signature, pointer_bindings, constexpr, values, element_bytes, file, first)
    body = function.body[1:] if function.body and isinstance(function.body[0], ast.Expr) and isinstance(function.body[0].value, ast.Constant) and isinstance(function.body[0].value.value, str) else function.body
    try:
        for statement in body:
            parser.statement(statement)
        expected = sum(_tl_call(node, "load") or _tl_call(node, "store") for node in ast.walk(function))
        if not parser.accesses:
            raise ParseFailure("Unsupported", "缺少显式 load/store 访问")
        # 有 constexpr 分支时 AST 总数包含未选择分支，因此只检查无分支函数的完整性。
        if not any(isinstance(node, ast.If) for node in ast.walk(function)) and expected != len(parser.accesses):
            raise ParseFailure("Unknown", "存在未建模的访存调用")
        if not any(point.kind == "store" for point in parser.accesses):
            raise ParseFailure("Unsupported", "缺少显式 store 访问")
        return ParseResult(
            "Supported",
            "全部绑定后访存访问点已归一化；不证明 Kernel 数值语义",
            signature,
            tuple(parser.accesses),
            False,
        )
    except ParseFailure as error:
        return ParseResult(error.status, error.reason, signature, tuple(parser.accesses))
