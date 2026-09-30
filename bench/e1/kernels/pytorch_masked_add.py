import torch
import triton
import triton.language as tl
from triton.language.extra.cuda.libdevice import tanh, rsqrt
debug = False
_CASTING_MODE_NONE: tl.constexpr = tl.constexpr(-1)
_CASTING_MODE_LLAMA: tl.constexpr = tl.constexpr(0)
_CASTING_MODE_GEMMA: tl.constexpr = tl.constexpr(1)

@triton.jit
def masked_add_kernel_with_bool_tensor(
    in_ptr0,
    in_ptr1,
    mask_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: "tl.constexpr",
):
    """Kernel that loads a bool tensor and uses it as a mask."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    valid = offsets < n_elements
    x = tl.load(in_ptr0 + offsets, mask=valid)
    y = tl.load(in_ptr1 + offsets, mask=valid)
    keep = tl.load(mask_ptr + offsets, mask=valid, other=0) != 0
    output = tl.where(keep, x + y, x)
    tl.store(out_ptr + offsets, output, mask=valid)
