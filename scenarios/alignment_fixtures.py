"""仅供第三阶段核对 tl.multiple_of 作用对象的两组可运行样例。"""

import torch
import triton
import triton.language as tl


@triton.jit
def pointer_hint(X, Y, N: tl.constexpr, B: tl.constexpr):
    base = tl.multiple_of(X, 16)
    idx = tl.program_id(0) * B + tl.arange(0, B)
    mask = idx < N
    value = tl.load(base + idx, mask=mask, other=0)
    tl.store(Y + idx, value + 1, mask=mask)


@triton.jit
def index_hint(X, Y, N: tl.constexpr, B: tl.constexpr):
    start = tl.multiple_of(tl.program_id(0) * B, 16)
    idx = start + tl.arange(0, B)
    mask = idx < N
    value = tl.load(X + idx, mask=mask, other=0)
    tl.store(Y + idx, value + 1, mask=mask)


def run_pointer_hint(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    n = x.shape[0]
    block = 128
    pointer_hint[(triton.cdiv(n, block),)](x, out, n, block)
    return out


def run_index_hint(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    n = x.shape[0]
    block = 128
    index_hint[(triton.cdiv(n, block),)](x, out, n, block)
    return out
