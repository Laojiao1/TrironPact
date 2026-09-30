import torch
import triton
import triton.language as tl
from triton.language.extra.cuda.libdevice import tanh, rsqrt
debug = False
_CASTING_MODE_NONE: tl.constexpr = tl.constexpr(-1)
_CASTING_MODE_LLAMA: tl.constexpr = tl.constexpr(0)
_CASTING_MODE_GEMMA: tl.constexpr = tl.constexpr(1)

@triton.jit
def softmax_kernel(X, Z, numel: tl.constexpr, shape: tl.constexpr, dim: tl.constexpr, ieee_rounding: tl.constexpr):
    # X is contiguous, so its row-major flat offsets are just an arange.
    offs = tl.arange(0, numel).reshape(shape)
    x = tl.load(X + offs)
    if ieee_rounding:
        z = x.softmax(dim=dim, ieee_rounding=True)
    else:
        z = tl.softmax(x, dim=dim)
    tl.static_assert(z.shape == x.shape, "softmax must preserve the input shape")
    tl.store(Z + offs, z)
