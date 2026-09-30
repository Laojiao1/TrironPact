import torch
import triton
import triton.language as tl
from triton.language.extra.cuda.libdevice import tanh, rsqrt
debug = False
_CASTING_MODE_NONE: tl.constexpr = tl.constexpr(-1)
_CASTING_MODE_LLAMA: tl.constexpr = tl.constexpr(0)
_CASTING_MODE_GEMMA: tl.constexpr = tl.constexpr(1)

@triton.jit
def triton_flip_kernel(out_ptr, x_ptr, N: tl.constexpr):
    pid = tl.program_id(0)
    x = tl.load(x_ptr + pid * N + tl.arange(0, N))
    shape: tl.constexpr = (N // 2, 2)
    y = x.reshape(shape)
    y = tl.flip(y, dim=1).reshape(x.shape)
    tl.store(out_ptr + pid * N + tl.arange(0, N), y)
