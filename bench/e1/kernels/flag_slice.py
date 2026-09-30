import torch
import triton
import triton.language as tl
from triton.language.extra.cuda.libdevice import tanh, rsqrt
debug = False
_CASTING_MODE_NONE: tl.constexpr = tl.constexpr(-1)
_CASTING_MODE_LLAMA: tl.constexpr = tl.constexpr(0)
_CASTING_MODE_GEMMA: tl.constexpr = tl.constexpr(1)

@triton.jit
def slice_kernel_2d(
    input_ptr,
    output_ptr,
    dim,
    start,
    step,
    input_stride_0,
    input_stride_1,
    output_size_0,
    output_size_1,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized 2D slice kernel."""
    pid_0 = tl.program_id(0)
    pid_1 = tl.program_id(1)

    out_idx_0 = pid_0
    out_idx_1_base = pid_1 * BLOCK_SIZE
    out_idx_1 = out_idx_1_base + tl.arange(0, BLOCK_SIZE)
    mask = out_idx_1 < output_size_1

    # Map to input coordinates
    in_idx_0 = tl.where(dim == 0, start + out_idx_0 * step, out_idx_0)
    in_idx_1 = tl.where(dim == 1, start + out_idx_1 * step, out_idx_1)

    input_offset = in_idx_0 * input_stride_0 + in_idx_1 * input_stride_1
    output_offset = out_idx_0 * output_size_1 + out_idx_1

    data = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    tl.store(output_ptr + output_offset, data, mask=mask)
