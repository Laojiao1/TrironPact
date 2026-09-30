import torch
import triton
import triton.language as tl
from triton.language.extra.cuda.libdevice import tanh, rsqrt
debug = False
_CASTING_MODE_NONE: tl.constexpr = tl.constexpr(-1)
_CASTING_MODE_LLAMA: tl.constexpr = tl.constexpr(0)
_CASTING_MODE_GEMMA: tl.constexpr = tl.constexpr(1)

@triton.jit
def double_strided_kernel(
    in_ptr,
    out_ptr,
    in_y_stride,
    out_y_stride,
    X_BLOCK_SIZE: "tl.constexpr",
    Y_BLOCK_SIZE: "tl.constexpr",
):
    xid = tl.program_id(axis=0)
    yid = tl.program_id(axis=1)
    x_start = xid * X_BLOCK_SIZE
    y_start = yid * Y_BLOCK_SIZE
    x_offsets = x_start + tl.arange(0, X_BLOCK_SIZE)
    y_offsets = y_start + tl.arange(0, Y_BLOCK_SIZE)
    src_offsets = y_offsets[:, None] * in_y_stride + x_offsets[None, :]
    dst_offsets = y_offsets[:, None] * out_y_stride + x_offsets[None, :]
    src = tl.load(in_ptr + src_offsets)
    tl.store(out_ptr + dst_offsets, src * 2.0)
