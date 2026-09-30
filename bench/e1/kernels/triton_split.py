import torch
import triton
import triton.language as tl
from triton.language.extra.cuda.libdevice import tanh, rsqrt
debug = False
_CASTING_MODE_NONE: tl.constexpr = tl.constexpr(-1)
_CASTING_MODE_LLAMA: tl.constexpr = tl.constexpr(0)
_CASTING_MODE_GEMMA: tl.constexpr = tl.constexpr(1)

@triton.jit
def kernel(X, Z1, Z2, N: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(X + offs)
    x1 = tl.reshape(x, (N // 2, 2))
    z1, z2 = tl.split(x1)
    tl.store(Z1 + tl.arange(0, N // 2), z1)
    tl.store(Z2 + tl.arange(0, N // 2), z2)
