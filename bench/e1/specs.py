"""e1 外部语义和有界调用规格；与候选规则、GuardPlan 注册分开。

正向候选表示具备可执行独立参考的研究样本，不表示解析器 Supported。
所有形状、grid 和 stride 单位为元素；指针直接传 Tensor，data_ptr 已含
视图 offset，启动时不得再次叠加 storage_offset。当前仅执行冻结的非空、
无输入输出 alias 的 float32/int32/bool CUDA smoke 域，不授予在线 Fast。
"""

from __future__ import annotations

import ast
import math
from dataclasses import dataclass
from typing import Literal

import torch
@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple[int, ...]
    dtype: Literal["float32", "int32", "bool"] = "float32"
    fill: Literal["random", "arange", "arange_inf", "empty"] = "random"


@dataclass(frozen=True)
class Binding:
    parameter: str
    expression: str


@dataclass(frozen=True)
class SmokeSpec:
    """输入/输出形状固定；bindings 必须覆盖原始签名，辅助统计量也核对。"""
    tensors: tuple[TensorSpec, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    bindings: tuple[Binding, ...]
    grid: tuple[int, ...]
    reference: str
    rtol: float = 1e-5
    atol: float = 1e-6


def value(expression: str, tensors: dict[str, torch.Tensor]) -> object:
    """求值受限 wrapper 实参；未知表达式抛错，不猜测绑定或静默降级。

    支持 Tensor 名、标量/元组常量、Tensor numel/size/stride；不执行 eval。
    """
    node = ast.parse(expression, mode="eval").body
    if isinstance(node, ast.Name) and node.id in tensors:
        return tensors[node.id]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
        tensor = tensors[node.func.value.id]
        if node.func.attr in {"stride", "size"} and len(node.args) == 1 and not node.keywords:
            axis = ast.literal_eval(node.args[0])
            if type(axis) is int:
                return getattr(tensor, node.func.attr)(axis)
        if node.func.attr == "numel" and not node.args and not node.keywords:
            return tensor.numel()
    return ast.literal_eval(node)


def reference(spec: SmokeSpec, tensors: dict[str, torch.Tensor]) -> tuple[torch.Tensor, ...]:
    """执行独立 PyTorch/Eager 数学语义；实现依据 catalog 中固定黄金参考。

    支持登记的纯值、广播、有限布局变换和行统计量。未知 reference 抛错，
    不从待测 Kernel 的输出推断真值；运行一致仅属有界经验观察。
    """
    args = [tensors[n] for n in spec.inputs]
    name = spec.reference
    x = args[0] if args else None
    if name == "add": return (args[0] + args[1],)
    if name == "sub": return (args[0] - args[1],)
    if name == "double": return (x * 2,)
    if name == "zeros": return (torch.zeros_like(x),)
    if name == "interleave":
        n = tensors[spec.outputs[0]].numel() // 2
        return (torch.stack((torch.arange(n, device="cuda", dtype=torch.int32), torch.arange(n, 2*n, device="cuda", dtype=torch.int32)), dim=-1).flatten(),)
    if name == "ravel": return (torch.arange(256, device="cuda", dtype=torch.int32),)
    if name == "broadcast": return (args[1].expand_as(x).clone(),)
    if name == "where_broadcast": return (torch.where(args[1], x, 0),)
    if name == "masked_add": return (torch.where(args[2], x + args[1], x),)
    if name == "cat": return (torch.cat(args),)
    if name == "transpose": return (x.T.contiguous(),)
    if name == "flip": return (torch.flip(x, (-1,)),)
    if name == "pair_flip": return (x.reshape(-1, 8, 2).flip(-1).reshape_as(x),)
    if name == "split": return (x.reshape(-1, 2)[:, 0].clone(), x.reshape(-1, 2)[:, 1].clone())
    if name == "slice2d": return (x[:, 2::2].contiguous(),)
    if name == "swiglu": return (torch.nn.functional.silu(x) * args[1],)
    if name == "fused_swiglu":
        a, b = x.chunk(2, dim=-1)
        return (torch.nn.functional.silu(a) * b,)
    if name == "geglu": return (torch.nn.functional.gelu(x, approximate="tanh") * args[1],)
    if name == "softmax": return (torch.softmax(x, dim=-1),)
    if name == "layernorm":
        # 黄金参考用 F.layer_norm；额外缓存输出用独立行统计量核对。
        mean = x.mean(-1)
        rstd = torch.rsqrt(x.var(-1, unbiased=False) + 1e-5)
        y = torch.nn.functional.layer_norm(x, (x.size(-1),), args[1], args[2], 1e-5)
        return tuple({"out": y, "mean": mean, "rstd": rstd}[n] for n in spec.outputs)
    if name in {"rmsnorm", "gemma_rmsnorm"}:
        rstd = torch.rsqrt(x.square().mean(-1) + 1e-6)
        y = x * rstd[:, None] * (args[1] + (1 if name == "gemma_rmsnorm" else 0))
        return tuple({"out": y, "rstd": rstd}[n] for n in spec.outputs)
    raise ValueError(f"Unknown：未登记的独立参考 {name}")


def construct(spec: SmokeSpec, seed: int) -> dict[str, torch.Tensor]:
    """构造冻结域的新分配 Tensor；无输入输出 alias，随机种子可追溯。"""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    tensors: dict[str, torch.Tensor] = {}
    for item in spec.tensors:
        dtype = getattr(torch, item.dtype)
        if item.fill in {"arange", "arange_inf"}:
            tensor = torch.arange(math.prod(item.shape), device="cuda", dtype=dtype).reshape(item.shape)
            if item.fill == "arange_inf":
                tensor.flatten()[-1] = float("inf")
        elif item.fill == "empty": tensor = torch.full(item.shape, False if dtype == torch.bool else -777, device="cuda", dtype=dtype)
        elif dtype == torch.bool: tensor = torch.randint(0, 2, item.shape, generator=generator, device="cuda").bool()
        else: tensor = torch.randn(item.shape, generator=generator, device="cuda").to(dtype)
        tensors[item.name] = tensor
    return tensors
