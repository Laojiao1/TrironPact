"""只审计本阶段已识别的 Python 启动语法，不执行 wrapper 或猜测动态调用。"""

from __future__ import annotations

import ast
import inspect
import textwrap

from pact.candidates import BindingResult, CallBinding


class _Unrecognized(Exception):
    pass


def _canon(node: ast.AST, env: dict[str, ast.AST], stack: frozenset[str] = frozenset()) -> str:
    if isinstance(node, ast.Name):
        if node.id in env:
            if node.id in stack:
                raise _Unrecognized("循环定义的调用参数")
            return _canon(env[node.id], env, stack | {node.id})
        return node.id
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return str(node.value)
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.attr in ("dtype", "device"):
        return f"{node.value.id}.{node.attr}"
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute) and isinstance(node.value.value, ast.Name) and node.value.attr == "shape" and isinstance(node.slice, ast.Constant) and type(node.slice.value) is int:
        return f"{node.value.value.id}.size({node.slice.value})"
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if isinstance(node.func.value, ast.Name) and node.func.attr in ("size", "stride") and len(node.args) == 1:
            return f"{node.func.value.id}.{node.func.attr}({_canon(node.args[0], env, stack)})"
        if isinstance(node.func.value, ast.Name) and node.func.attr == "numel" and not node.args:
            return f"{node.func.value.id}.numel()"
        if isinstance(node.func.value, ast.Name) and node.func.value.id == "triton" and node.func.attr == "next_power_of_2" and len(node.args) == 1:
            return f"next_power_of_2({_canon(node.args[0], env, stack)})"
        if isinstance(node.func.value, ast.Name) and node.func.value.id == "triton" and node.func.attr == "cdiv" and len(node.args) == 2:
            return f"ceil({_canon(node.args[0], env, stack)}/{_canon(node.args[1], env, stack)})"
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        return f"{_canon(node.left, env, stack)}*{_canon(node.right, env, stack)}"
    if isinstance(node, ast.Tuple):
        return ",".join(_canon(item, env, stack) for item in node.elts)
    raise _Unrecognized(f"未识别调用表达式：{ast.unparse(node)}")


def audit_call_site(wrapper, kernel_name: str, binding: CallBinding) -> BindingResult:
    """核对实参、grid、输出形状和 dtype 来源；不解析任意 Python 副作用。"""
    try:
        source = textwrap.dedent(inspect.getsource(wrapper))
        fn = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef))
    except (OSError, TypeError, SyntaxError, StopIteration):
        return BindingResult("Unknown", ("无法读取可信的 wrapper 源码",))
    calls = [node for node in ast.walk(fn) if isinstance(node, ast.Call) and isinstance(node.func, ast.Subscript) and isinstance(node.func.value, ast.Name) and node.func.value.id == kernel_name]
    if len(calls) != 1:
        return BindingResult("Unknown", ("启动调用缺失或存在多个匹配点",))
    launch = calls[0]
    if not isinstance(launch.func.slice, ast.Tuple) or len(launch.func.slice.elts) != 1:
        return BindingResult("Unknown", ("只支持单轴 grid",))
    branches = [node for node in ast.walk(fn) if isinstance(node, ast.If) and any(child is launch for statement in node.body for child in ast.walk(statement))]
    scope = min(branches, key=lambda node: sum(1 for statement in node.body for _ in ast.walk(statement))).body if branches else fn.body
    env: dict[str, ast.AST] = {}
    outputs: list[ast.Call] = []
    late_or_redefined = False
    for node in (child for statement in scope for child in ast.walk(statement)):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if node.lineno >= launch.lineno:
            late_or_redefined = True
        if isinstance(target, ast.Tuple) and isinstance(node.value, ast.Attribute) and node.value.attr == "shape" and isinstance(node.value.value, ast.Name):
            for axis, name in enumerate(target.elts):
                if isinstance(name, ast.Name):
                    if name.id in env:
                        late_or_redefined = True
                    env[name.id] = ast.parse(f"{node.value.value.id}.shape[{axis}]", mode="eval").body
        elif isinstance(target, ast.Name):
            if target.id == "out" and isinstance(node.value, ast.Call):
                outputs.append(node.value)
            elif target.id in env:
                late_or_redefined = True
            else:
                env[target.id] = node.value
    if len(outputs) != 1:
        return BindingResult("Unknown", ("新建输出分配缺失或不唯一",))
    reasons = []
    if late_or_redefined:
        reasons.append("调用参数或输出存在启动后的赋值或重复定义")
    if not any(isinstance(node, ast.Return) and isinstance(node.value, ast.Name) and node.value.id == "out" and node.lineno > launch.lineno for node in ast.walk(fn)):
        reasons.append("wrapper 未返回已核对的新建输出")
    try:
        if len(launch.args) != len(binding.pointers) + len(binding.scalars):
            reasons.append("启动实参数量与绑定不符")
        signature = [(param, "out" if tensor == "OUT" else tensor.lower()) for param, tensor in binding.pointers]
        signature += [(param, value.split(" = ", 1)[0]) for param, value in binding.scalars]
        for (param, expected), actual in zip(signature, launch.args):
            observed = _canon(actual, env)
            if observed != expected:
                for name, value in binding.scalars:
                    if value.startswith("x.size(") or value.startswith("y.size("):
                        observed = observed.replace(value, name)
            if observed != expected:
                reasons.append(f"实参 {param} 与声明不符")
        grid_node = launch.func.slice.elts[0]
        actual_grid = _canon(grid_node, env)
        declared = binding.grid
        # 将 grid 中绑定的标量名展开为对应实参表达式的符号名称。
        scalar_names = dict(binding.scalars)
        for name, value in sorted(scalar_names.items(), key=lambda item: len(item[1]), reverse=True):
            actual_grid = actual_grid.replace(value.split(" = ", 1)[0], name)
        if actual_grid != declared:
            reasons.append("实际 grid 与声明不符")
        output = outputs[0]
        if not isinstance(output.func, ast.Attribute) or not isinstance(output.func.value, ast.Name) or output.func.value.id != "torch":
            reasons.append("输出不是受支持的 torch 分配")
        elif output.func.attr == "empty":
            shape = tuple(_canon(item, env) for item in output.args[0].elts) if len(output.args) == 1 and isinstance(output.args[0], ast.Tuple) else ()
            expected_shape = tuple(dict(binding.scalars).get(item, item) for item in binding.output_shape)
            if shape != expected_shape:
                reasons.append("实际输出形状与声明不符")
            kwargs = {item.arg: _canon(item.value, env) for item in output.keywords}
            if kwargs.get("dtype") != "x.dtype" or kwargs.get("device") != "x.device":
                reasons.append("输出 dtype/device 未绑定输入")
        elif output.func.attr == "empty_like":
            if len(output.args) != 1 or _canon(output.args[0], env) != "x" or len(binding.output_shape) != 1:
                reasons.append("empty_like 的输入或输出维度不符")
            kwargs = {item.arg: ast.unparse(item.value) for item in output.keywords}
            if kwargs.get("memory_format") != "torch.contiguous_format":
                reasons.append("输出连续布局未核对")
        else:
            reasons.append("输出分配形式未支持")
    except (_Unrecognized, AttributeError, IndexError) as error:
        reasons.append(str(error))
    return BindingResult("Unknown" if reasons else "Supported", tuple(reasons))
