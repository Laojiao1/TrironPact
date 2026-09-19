"""本文件复现 Triton 官方向量加法教程的访存结构，用作外部源码适配案例。

来源：https://github.com/triton-lang/triton/blob/main/python/tutorials/01-vector-add.py
本地仅保留 Kernel 与最小启动函数，注释改用中文；它是教程案例，不是生产缺陷案例。

"""

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def run_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    block = 128
    add_kernel[(triton.cdiv(x.numel(), block),)](x, y, out, x.numel(), block)
    return out
