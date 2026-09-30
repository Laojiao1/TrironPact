import torch
import triton
import triton.language as tl
from triton.language.extra.cuda.libdevice import tanh, rsqrt
debug = False
_CASTING_MODE_NONE: tl.constexpr = tl.constexpr(-1)
_CASTING_MODE_LLAMA: tl.constexpr = tl.constexpr(0)
_CASTING_MODE_GEMMA: tl.constexpr = tl.constexpr(1)

@triton.jit
def triton_ravel(out_ptr):
    a = tl.arange(0, 256)
    a = tl.reshape(a, (32, 8))
    a = tl.ravel(a)
    tl.store(out_ptr + tl.arange(0, 256), a)
