import torch
import triton
import triton.language as tl
from triton.language.extra.cuda.libdevice import tanh, rsqrt
debug = False
_CASTING_MODE_NONE: tl.constexpr = tl.constexpr(-1)
_CASTING_MODE_LLAMA: tl.constexpr = tl.constexpr(0)
_CASTING_MODE_GEMMA: tl.constexpr = tl.constexpr(1)

@triton.jit
def kernel(In, Out, in_shape1: tl.constexpr, in_shape2: tl.constexpr, ou_shape1: tl.constexpr,
           ou_shape2: tl.constexpr, trans1: tl.constexpr, trans2: tl.constexpr):
    in_offs = tl.arange(0, in_shape1)[:, None] * in_shape2 + tl.arange(0, in_shape2)[None, :]
    ou_offs = tl.arange(0, ou_shape1)[:, None] * ou_shape2 + tl.arange(0, ou_shape2)[None, :]
    tl.store(Out + ou_offs, tl.permute(tl.load(In + in_offs), (trans1, trans2)))
