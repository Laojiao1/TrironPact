import torch
import triton
import triton.language as tl
from triton.language.extra.cuda.libdevice import tanh, rsqrt
debug = False
_CASTING_MODE_NONE: tl.constexpr = tl.constexpr(-1)
_CASTING_MODE_LLAMA: tl.constexpr = tl.constexpr(0)
_CASTING_MODE_GEMMA: tl.constexpr = tl.constexpr(1)

@triton.jit
def flip_kernel(X, Z, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, dim: tl.constexpr):
    offx = tl.arange(0, M) * N * K
    offy = tl.arange(0, N) * K
    offz = tl.arange(0, K)
    off3d = offx[:, None, None] + offy[None, :, None] + offz[None, None, :]
    x = tl.load(X + off3d)
    x = tl.flip(x, dim)
    tl.store(Z + off3d, x)
