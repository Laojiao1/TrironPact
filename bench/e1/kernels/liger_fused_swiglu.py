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
def _swiglu_fused_gate_up_forward_kernel(
    y_ptr, c_ptr, in_stride, out_stride, ffn_size: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    program_id = tl.program_id(0).to(tl.int64)

    y_ptr += program_id * in_stride
    c_ptr += program_id * out_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < ffn_size

    # Gate occupies columns [0, ffn_size), up occupies [ffn_size, 2 * ffn_size) of the same row.
    gate = tl.load(y_ptr + col_offsets, mask=mask, other=0).to(tl.float32)
    up = tl.load(y_ptr + ffn_size + col_offsets, mask=mask, other=0)
    tl.store(c_ptr + col_offsets, silu(gate).cast(up.dtype) * up, mask=mask)
