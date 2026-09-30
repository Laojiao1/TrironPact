import torch
import triton
import triton.language as tl
from triton.language.extra.cuda.libdevice import tanh, rsqrt
debug = False
_CASTING_MODE_NONE: tl.constexpr = tl.constexpr(-1)
_CASTING_MODE_LLAMA: tl.constexpr = tl.constexpr(0)
_CASTING_MODE_GEMMA: tl.constexpr = tl.constexpr(1)

@triton.jit
def where_kernel(cond_ptr, a_ptr, out_ptr, BLOCK_SIZE: tl.constexpr):
    xoffsets = tl.arange(0, BLOCK_SIZE)[:, None]
    yoffsets = tl.arange(0, BLOCK_SIZE)[None, :]

    mask = tl.load(cond_ptr + yoffsets)
    vals = tl.load(a_ptr + yoffsets + BLOCK_SIZE * xoffsets)
    res = tl.where(mask, vals, 0.)
    tl.store(out_ptr + yoffsets + BLOCK_SIZE * xoffsets, res)
