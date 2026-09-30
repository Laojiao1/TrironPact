import torch
import triton
import triton.language as tl
from triton.language.extra.cuda.libdevice import tanh, rsqrt
debug = False
_CASTING_MODE_NONE: tl.constexpr = tl.constexpr(-1)
_CASTING_MODE_LLAMA: tl.constexpr = tl.constexpr(0)
_CASTING_MODE_GEMMA: tl.constexpr = tl.constexpr(1)

@triton.jit(debug=debug)
def kernel(Z, N: tl.constexpr):
    z = tl.interleave(tl.arange(0, N), tl.arange(N, 2 * N))
    tl.store(Z + tl.arange(0, 2 * N), z)
