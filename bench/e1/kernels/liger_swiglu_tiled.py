import torch
import triton
import triton.language as tl
from triton.language.extra.cuda.libdevice import tanh, rsqrt
debug = False
_CASTING_MODE_NONE: tl.constexpr = tl.constexpr(-1)
_CASTING_MODE_LLAMA: tl.constexpr = tl.constexpr(0)
_CASTING_MODE_GEMMA: tl.constexpr = tl.constexpr(1)

@triton.jit
def silu(x):
    return x * tl.sigmoid(x)

@triton.jit
def _swiglu_forward_kernel_tiled(a_ptr, b_ptr, c_ptr, stride, gate_multiplier, n_cols, BLOCK_SIZE: tl.constexpr):
    # Blackwell path: 2D grid -- axis 0 selects the row, axis 1 the column tile.
    program_id = tl.program_id(0).to(tl.int64)
    col_tile = tl.program_id(1)

    # locate start index (row base; column offset added below)
    a_ptr += program_id * stride
    b_ptr += program_id * stride
    c_ptr += program_id * stride

    col_offsets = col_tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # sigmoid requires type float32
    a_row = tl.load(a_ptr + col_offsets, mask=mask, other=0).to(tl.float32) * gate_multiplier.to(tl.float32)
    b_row = tl.load(b_ptr + col_offsets, mask=mask, other=0)
    c_row = silu(a_row).cast(b_row.dtype) * b_row
    tl.store(c_ptr + col_offsets, c_row, mask=mask)
