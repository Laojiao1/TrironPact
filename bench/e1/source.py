"""原始资产中的函数提取、语法特征和改名不敏感去重。"""

from __future__ import annotations

import ast
import hashlib
import textwrap
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FunctionSource:
    source: str
    line: int
    node: ast.FunctionDef


def extract(path: Path, qualified_name: str) -> FunctionSource:
    """按嵌套限定名提取原始函数（含 decorator），不改写函数体。

    仅缩进归零以便独立加载；不存在或不唯一时抛出 ValueError。
    返回的行号仍指向原始文件，不能以独立加载文件的行号替代来源。
    """
    content = path.read_text(encoding="utf-8")
    matches: list[ast.FunctionDef] = []

    def walk(node: ast.AST, prefix: str = "") -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.ClassDef)):
                name = prefix + child.name
                if name == qualified_name and isinstance(child, ast.FunctionDef):
                    matches.append(child)
                walk(child, name + ".")
            else:
                walk(child, prefix)

    walk(ast.parse(content))
    if len(matches) != 1:
        raise ValueError(f"函数来源不唯一：{path}:{qualified_name}")
    node = matches[0]
    first = min([node.lineno, *(d.lineno for d in node.decorator_list)])
    source = textwrap.dedent("\n".join(content.splitlines()[first - 1:node.end_lineno]))
    return FunctionSource(source, first, ast.parse(source).body[0])


def body_fingerprint(source: str) -> str:
    """忽略函数名、局部变量名、注释和 decorator；保留运算、常量与访存。

    同名改写和仅 decorator 配置不同的副本不可重复计数。
    本指纹不是程序语义等价判定，模板相似性仍须人工审计。
    """
    node = ast.parse(source).body[0]
    names: dict[str, str] = {}

    class Normalize(ast.NodeTransformer):
        def visit_Name(self, item: ast.Name) -> ast.AST:
            if item.id not in {"tl", "triton", "torch"}:
                item.id = names.setdefault(item.id, f"v{len(names)}")
            return item

        def visit_arg(self, item: ast.arg) -> ast.AST:
            item.arg = names.setdefault(item.arg, f"v{len(names)}")
            item.annotation = None
            return item

    node.name = "kernel"
    node.decorator_list = []
    if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
        node.body.pop(0)
    Normalize().visit(node)
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def features(source: str) -> dict:
    """返回语法事实；它们不证明访问合法性或 Fast 可行性。"""
    tree = ast.parse(source)
    nodes = list(ast.walk(tree))
    calls = [ast.unparse(n.func) for n in nodes if isinstance(n, ast.Call)]
    axes = sorted({int(n.args[0].value) if n.args else int(n.keywords[0].value.value)
                   for n in nodes if isinstance(n, ast.Call) and ast.unparse(n.func) == "tl.program_id"
                   and ((n.args and isinstance(n.args[0], ast.Constant)) or (n.keywords and isinstance(n.keywords[0].value, ast.Constant)))})
    return {"program_id_axes": axes, "load_count": calls.count("tl.load"), "store_count": calls.count("tl.store"),
            "explicit_masks": sum(isinstance(n, ast.keyword) and n.arg == "mask" for n in nodes),
            "loop": any(isinstance(n, (ast.For, ast.While)) for n in nodes),
            "branch": any(isinstance(n, ast.If) for n in nodes),
            "reduction": any(c in {"tl.sum", "tl.max", "tl.min", "tl.reduce", "tl.softmax"} or c.endswith(".softmax") for c in calls),
            "cast": any(c.endswith((".to", ".cast")) for c in calls),
            "pointer_update": any(isinstance(n, ast.AugAssign) for n in nodes),
            "broadcast": any(isinstance(n, ast.Constant) and n.value is None for n in nodes),
            "layout_transform": any(c in {"tl.permute", "tl.trans", "tl.reshape", "tl.flip", "tl.cat", "tl.split", "tl.ravel", "tl.interleave"} for c in calls)}
