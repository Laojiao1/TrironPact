"""保存自研 A/A2/B Kernel，并统一调用教程向量加法案例。"""

import torch
import triton
import triton.language as tl

from scenarios.external_add import run_add


@triton.jit
def flat_2d(X, Y, M: tl.constexpr, N: tl.constexpr, B: tl.constexpr):
    # 程序员假定矩阵是连续存储的，直接用一维扁平索引取数据
    idx = tl.program_id(0) * B + tl.arange(0, B)
    mask = idx < M * N
    value = tl.load(X + idx, mask=mask, other=0) 
    tl.store(Y + idx, value + 1, mask=mask)


@triton.jit
def row_stride_2d(X, Y, M: tl.constexpr, N: tl.constexpr, S0: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    mask = col < N
    # 行距由输入传入；列内仍按相邻元素读取。
    value = tl.load(X + row * S0 + col, mask=mask, other=0)
    tl.store(Y + row * N + col, value + 1, mask=mask)


@triton.jit
def masked_1d(X, Y, N: tl.constexpr, B: tl.constexpr):
    idx = tl.program_id(0) * B + tl.arange(0, B)
    mask = idx < N
    # 最后一块不足 B 个元素时，mask 排除多余地址。
    value = tl.load(X + idx, mask=mask, other=0)
    tl.store(Y + idx, value + 1, mask=mask)


KERNELS = {"A": flat_2d, "A2": row_stride_2d, "B": masked_1d}


def run_fast(name: str, x: torch.Tensor, y: torch.Tensor | None = None) -> torch.Tensor:
    """调用指定快路径；此处不做安全判断，调用方必须先检查契约。"""
    if name == "D":
        if y is None:
            raise ValueError("向量加法缺少第二个输入")
        return run_add(x, y)
    if name == "A":
        m, n = x.shape
        out = torch.empty((m, n), dtype=x.dtype, device=x.device)
        flat_2d[(triton.cdiv(m * n, 128),)](x, out, m, n, 128)
    elif name == "A2":
        m, n = x.shape
        out = torch.empty((m, n), dtype=x.dtype, device=x.device)
        block = triton.next_power_of_2(n)
        row_stride_2d[(m,)](x, out, m, n, x.stride(0), block)
    elif name == "B":
        (n,) = x.shape
        out = torch.empty((n,), dtype=x.dtype, device=x.device)
        masked_1d[(triton.cdiv(n, 128),)](x, out, n, 128)
    else:
        raise ValueError(f"未知案例：{name}")
    return out
