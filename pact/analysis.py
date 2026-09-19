"""从 Triton AST 提取 Access IR、访存跨度、mask 与布局候选谓词"""

import ast
import copy
import inspect
import textwrap
from dataclasses import asdict, dataclass
from pathlib import Path

from scenarios.kernels import KERNELS
from scenarios.external_add import add_kernel
from pact.semantics import OperatorMeaning, SEMANTICS


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
    semantics: OperatorMeaning
    accesses: tuple[Access, ...]
    predicates: tuple[Predicate, ...]
    mask_excludes_tail: bool
    refinement_state: str = "尚未发生动态候选修改"

    def to_dict(self) -> dict:
        return asdict(self)


def _same(a: ast.AST, b: ast.AST) -> bool:
    return ast.dump(a, include_attributes=False) == ast.dump(b, include_attributes=False)


def _name(node: ast.AST, value: str) -> bool:
    return isinstance(node, ast.Name) and node.id == value


def _call(node: ast.AST, func: str) -> bool:
    return isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name) and node.func.value.id == "tl" and node.func.attr == func


def _call_annotation(node: ast.Attribute) -> bool:
    return isinstance(node.value, ast.Name) and node.value.id == "tl" and node.attr == "constexpr"


def _source_location(fn) -> tuple[str, int, str]:
    """AST 行号从函数开头计数；转换成可直接打开的源码文件行号。"""
    lines, first_line = inspect.getsourcelines(fn)
    source_file = Path(inspect.getsourcefile(fn)).resolve().relative_to(Path(__file__).resolve().parent.parent).as_posix()
    return textwrap.dedent("".join(lines)), first_line, source_file


def _sum_parts(node: ast.AST) -> list[ast.AST]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _sum_parts(node.left) + _sum_parts(node.right)
    return [node]


# 从指针算术表达式（如 X + idx）中提取出基址指针 X 和偏移量表达式 idx
def _pointer_parts(node: ast.AST, base: str) -> list[ast.AST] | None:
    parts = _sum_parts(node)
    if not parts or not _name(parts[0], base):
        return None
    # 提取偏移量公式 offsets，一般来说 parts[0] 是基地址
    return parts[1:]


# 将 Kernel 内部所有的临时变量赋值（如 idx = pid * B + lane）向前递归展开到实际的 tl.load 和 tl.store
def _expand(node: ast.AST, assignments: dict[str, ast.AST], seen: frozenset[str] = frozenset()) -> ast.AST:
    """只展开当前函数内的简单赋值，保留 Triton 索引调用。"""
    if isinstance(node, ast.Name) and node.id in assignments and node.id not in seen:
        # 递归找表达式
        return _expand(assignments[node.id], assignments, seen | {node.id})
    result = copy.deepcopy(node)
    for field, value in ast.iter_fields(result):
        if isinstance(value, ast.AST):
            setattr(result, field, _expand(value, assignments, seen))
        elif isinstance(value, list):
            setattr(result, field, [_expand(item, assignments, seen) if isinstance(item, ast.AST) else item for item in value])
    return result


def _mul_names(node: ast.AST, a: str, b: str) -> bool:
    return isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult) and ((_name(node.left, a) and _name(node.right, b)) or (_name(node.left, b) and _name(node.right, a)))


def _is_program_id(node: ast.AST) -> bool:
    if not _call(node, "program_id"):
        return False
    if len(node.args) == 1 and isinstance(node.args[0], ast.Constant) and node.args[0].value == 0:
        return True
    return not node.args and len(node.keywords) == 1 and node.keywords[0].arg == "axis" and isinstance(node.keywords[0].value, ast.Constant) and node.keywords[0].value.value == 0


def _is_arange(node: ast.AST, block_name: str = "B") -> bool:
    return _call(node, "arange") and len(node.args) == 2 and isinstance(node.args[0], ast.Constant) and node.args[0].value == 0 and _name(node.args[1], block_name)


def _is_linear_index(node: ast.AST, assignments: dict[str, ast.AST], block_name: str = "B") -> bool:
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Add):
        return False
    parts = (node.left, node.right)
    for product, lane in (parts, parts[::-1]):
        if not isinstance(product, ast.BinOp) or not isinstance(product.op, ast.Mult):
            continue
        lane_expr = assignments.get(lane.id) if isinstance(lane, ast.Name) else lane
        if not _is_arange(lane_expr, block_name):
            continue
        for pid, block in ((product.left, product.right), (product.right, product.left)):
            pid_expr = assignments.get(pid.id) if isinstance(pid, ast.Name) else pid
            if _name(block, block_name) and _is_program_id(pid_expr):
                return True
    return False


def _comparison(node: ast.AST, left: str, right: ast.AST) -> bool:
    return isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Lt) and len(node.comparators) == 1 and _name(node.left, left) and _same(node.comparators[0], right)


# 自动推导契约
def _unsupported(name: str, reason: str, accesses: list[Access] | None = None) -> Analysis:
    return Analysis(name, "Unsupported", reason, SEMANTICS[name], tuple(accesses or ()), (), False)


def _predicate(name: str, accesses: list[Access], tensor: str, dimension: int, expected: str, source: str) -> Predicate:
    """让每条布局条件都指向实际 load 和外部声明的逻辑读取。"""
    meaning = SEMANTICS[name].input_for(tensor)
    access = next(item for item in accesses if item.kind == "load" and item.tensor == meaning.pointer)
    return Predicate(dimension, expected, "语义", source, access.line, access.source_file, meaning.pointer, meaning.logical_read, tensor=tensor)


def extract(name: str) -> Analysis:
    """仅识别当前四个受限变体；不能解释的语法明确拒绝。"""
    if name == "D":
        return _extract_external_add()
    if name not in KERNELS:
        raise ValueError(f"未知案例：{name}")
    source, first_line, source_file = _source_location(KERNELS[name].fn)
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
    if sum(_call(node, "load") for node in calls) != 1 or sum(_call(node, "store") for node in calls) != 1:
        return _unsupported(name, "只支持各一个 load/store 访存点")
    if any(_call(node, "multiple_of") for node in calls):
        return _unsupported(name, "尚未建模 multiple_of 的实际表达式")
    constexpr = tuple(arg.arg for arg in function.args.args if isinstance(arg.annotation, ast.Attribute) and _call_annotation(arg.annotation))
    assignments: dict[str, ast.AST] = {}
    load_call = store_call = None
    accesses: list[Access] = []

    for statement in function.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
            assignments[statement.targets[0].id] = statement.value
            if _call(statement.value, "load"):
                load_call = statement.value
        elif isinstance(statement, ast.Expr) and _call(statement.value, "store"):
            store_call = statement.value
        else:
            return _unsupported(name, f"不支持第 {statement.lineno} 行的语句")

    if load_call is None or store_call is None:
        return _unsupported(name, "需要各一个显式 load 和 store")
    load_mask = next((kw.value for kw in load_call.keywords if kw.arg == "mask"), None)
    store_mask = next((kw.value for kw in store_call.keywords if kw.arg == "mask"), None)
    if not _name(load_mask, "mask") or not _name(store_mask, "mask"):
        return _unsupported(name, "load/store 必须使用同一个显式 mask")
    mask = assignments.get("mask")
    if mask is None:
        return _unsupported(name, "没有找到 mask 定义")

    for kind, call, base in (("load", load_call, "X"), ("store", store_call, "Y")):
        ptr = call.args[0]
        parts = _pointer_parts(ptr, base)
        if parts is None:
            return _unsupported(name, f"{kind} 指针没有以 {base} 为基址")
        expanded = [_expand(part, assignments) for part in parts]
        symbols = tuple(sorted({item.id for part in expanded for item in ast.walk(part) if isinstance(item, ast.Name)}))
        index_calls = tuple(sorted({item.func.attr for part in expanded for item in ast.walk(part) if _call(item, "program_id") or _call(item, "arange")}))
        accesses.append(Access(kind, base, " + ".join(ast.unparse(part) for part in parts), " + ".join(ast.unparse(part) for part in expanded), ast.unparse(mask), 4, first_line + call.lineno - 1, source_file, symbols, index_calls, constexpr))

    load_parts = _pointer_parts(load_call.args[0], "X")
    store_parts = _pointer_parts(store_call.args[0], "Y")
    # 地址符合模板还不够，输出计算也要符合输入的参考语义。
    if not isinstance(store_call.args[1], ast.BinOp) or not isinstance(store_call.args[1].op, ast.Add) or not _name(store_call.args[1].left, "value") or not isinstance(store_call.args[1].right, ast.Constant) or store_call.args[1].right.value != 1:
        return _unsupported(name, "输出计算不符合声明的逐元素加一语义", accesses)
    if assignments.get("value") is not load_call:
        return _unsupported(name, "load 结果与输出计算没有对应", accesses)

    if len(load_parts) == len(store_parts) == 1 and _name(load_parts[0], "idx") and _same(load_parts[0], store_parts[0]):
        # 一维线性索引与二维行优先索引共享这个地址模板。
        if not _is_linear_index(assignments.get("idx"), assignments):
            return _unsupported(name, "线性索引不符合 pid*B+lane 模板", accesses)
        bound = ast.parse("M * N" if name == "A" else "N", mode="eval").body
        if not _comparison(mask, "idx", bound):
            return _unsupported(name, "线性尾部 mask 无法证明覆盖边界", accesses)
        if name == "A":
            predicates = (
                _predicate(name, accesses, "X", 0, "size(1)", "X + idx 与二维行优先逻辑索引对应"),
                _predicate(name, accesses, "X", 1, "1", "X + idx 的相邻列地址差为 1"),
            )
        elif name == "B":
            predicates = (_predicate(name, accesses, "X", 0, "1", "X + idx 的相邻元素地址差为 1"),)
        else:
            return _unsupported(name, "语义维度与线性模板不匹配", accesses)
    elif name == "A2" and len(load_parts) == len(store_parts) == 2:
        if not (_mul_names(load_parts[0], "row", "S0") and _name(load_parts[1], "col") and _mul_names(store_parts[0], "row", "N") and _name(store_parts[1], "col")):
            return _unsupported(name, "行步长地址模板不匹配", accesses)
        if not _is_program_id(assignments.get("row")) or not _is_arange(assignments.get("col")) or not _comparison(mask, "col", ast.Name(id="N", ctx=ast.Load())):
            return _unsupported(name, "行列索引或尾部 mask 不匹配", accesses)
        predicates = (_predicate(name, accesses, "X", 1, "1", "X + row*S0 + col 的相邻列地址差为 1；S0 来自输入 stride(0)"),)
    else:
        return _unsupported(name, "访存地址不属于受支持模板", accesses)

    return Analysis(name, "Supported", "在声明的算子语义、启动方式和受支持语法内完成静态提取", SEMANTICS[name], tuple(accesses), predicates, True)


def _extract_external_add() -> Analysis:
    """独立检查官方教程的双输入结构，不能从名称直接填入契约。"""
    name = "D"
    source, first_line, source_file = _source_location(add_kernel.fn)
    function = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef))
    calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
    if sum(_call(node, "load") for node in calls) != 2 or sum(_call(node, "store") for node in calls) != 1:
        return _unsupported(name, "官方案例需要两个 load、一个 store")
    if any(_call(node, "multiple_of") for node in calls):
        return _unsupported(name, "尚未建模优化提示")
    assignments: dict[str, ast.AST] = {}
    store = None
    for statement in function.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1 and isinstance(statement.targets[0], ast.Name):
            assignments[statement.targets[0].id] = statement.value
        elif isinstance(statement, ast.Expr) and _call(statement.value, "store"):
            store = statement.value
        else:
            return _unsupported(name, f"不支持第 {statement.lineno} 行的语句")
    if store is None or not _is_linear_index(_expand(ast.Name(id="offsets", ctx=ast.Load()), assignments), {}, "BLOCK_SIZE"):
        return _unsupported(name, "线性索引不符合受支持模板")
    if not _comparison(assignments.get("mask"), "offsets", ast.Name(id="n_elements", ctx=ast.Load())):
        return _unsupported(name, "尾部 mask 不符合受支持模板")
    if not isinstance(assignments.get("output"), ast.BinOp) or not isinstance(assignments["output"].op, ast.Add) or not _name(assignments["output"].left, "x") or not _name(assignments["output"].right, "y") or not _name(store.args[1], "output"):
        return _unsupported(name, "输出计算不符合声明的向量加法语义")

    constexpr = ("BLOCK_SIZE",)
    accesses: list[Access] = []
    for kind, result_name, base in (("load", "x", "x_ptr"), ("load", "y", "y_ptr"), ("store", "", "output_ptr")):
        call = store if kind == "store" else assignments.get(result_name)
        if call is None or (kind == "load" and not _call(call, "load")):
            return _unsupported(name, f"没有找到 {base} 的 load/store")
        parts = _pointer_parts(call.args[0], base)
        mask = next((kw.value for kw in call.keywords if kw.arg == "mask"), None)
        if parts is None or len(parts) != 1 or not _name(parts[0], "offsets") or not _name(mask, "mask"):
            return _unsupported(name, f"{base} 的地址或 mask 不受支持")
        expanded = _expand(parts[0], assignments)
        symbols = tuple(sorted({item.id for item in ast.walk(expanded) if isinstance(item, ast.Name)}))
        accesses.append(Access(kind, base, "offsets", ast.unparse(expanded), "offsets < n_elements", 4, first_line + call.lineno - 1, source_file, symbols, ("arange", "program_id"), constexpr))
    predicates = (
        _predicate(name, accesses, "X", 0, "1", "x_ptr + offsets 对应 x 的逐元素索引"),
        _predicate(name, accesses, "Y", 0, "1", "y_ptr + offsets 对应 y 的逐元素索引"),
    )
    return Analysis(name, "Supported", "对 Triton 官方向量加法教程的两个输入访存点完成提取", SEMANTICS[name], tuple(accesses), predicates, True, "外部教程案例，未发生动态修订")
