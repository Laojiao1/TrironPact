"""第六阶段离线基准 Kernel；不注册到在线 Guard 或分派入口。"""

import triton
import triton.language as tl
import torch


@triton.jit
def relu_1d(X, Y, N: tl.constexpr, B: tl.constexpr):
    idx = tl.program_id(0) * B + tl.arange(0, B)
    mask = idx < N
    value = tl.load(X + idx, mask=mask, other=0.0)
    result = tl.maximum(value, 0.0)
    tl.store(Y + idx, result, mask=mask)


@triton.jit
def square_1d(X, Y, N: tl.constexpr, B: tl.constexpr):
    idx = tl.program_id(0) * B + tl.arange(0, B)
    mask = idx < N
    value = tl.load(X + idx, mask=mask, other=0.0)
    result = value * value
    tl.store(Y + idx, result, mask=mask)


@triton.jit
def multiply_1d(X, Z, Y, N: tl.constexpr, B: tl.constexpr):
    idx = tl.program_id(0) * B + tl.arange(0, B)
    mask = idx < N
    left = tl.load(X + idx, mask=mask, other=0.0)
    right = tl.load(Z + idx, mask=mask, other=0.0)
    result = left * right
    tl.store(Y + idx, result, mask=mask)


@triton.jit
def axpy_1d(X, Z, Y, ALPHA: tl.constexpr, N: tl.constexpr, B: tl.constexpr):
    idx = tl.program_id(0) * B + tl.arange(0, B)
    mask = idx < N
    left = tl.load(X + idx, mask=mask, other=0.0)
    right = tl.load(Z + idx, mask=mask, other=0.0)
    result = ALPHA * left + right
    tl.store(Y + idx, result, mask=mask)


@triton.jit
def feature_scale_2d(X, F, Y, M: tl.constexpr, N: tl.constexpr, S0: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    mask = col < N
    value = tl.load(X + row * S0 + col, mask=mask, other=0.0)
    feature = tl.load(F + col, mask=mask, other=0.0)
    result = value * feature
    tl.store(Y + row * N + col, result, mask=mask)


@triton.jit
def feature_bias_2d(X, F, Y, M: tl.constexpr, N: tl.constexpr, S0: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    mask = col < N
    value = tl.load(X + row * S0 + col, mask=mask, other=0.0)
    feature = tl.load(F + col, mask=mask, other=0.0)
    result = value + feature
    tl.store(Y + row * N + col, result, mask=mask)


def run_relu(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    block = 128
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    relu_1d[(triton.cdiv(n, block),)](x, out, n, block)
    return out


def run_square(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    block = 128
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    square_1d[(triton.cdiv(n, block),)](x, out, n, block)
    return out


def run_multiply(x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    block = 128
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    multiply_1d[(triton.cdiv(n, block),)](x, z, out, n, block)
    return out


def run_axpy(x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    block = 128
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    axpy_1d[(triton.cdiv(n, block),)](x, z, out, 2, n, block)
    return out


def run_feature_scale(x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block = triton.next_power_of_2(n)
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    feature_scale_2d[(m,)](x, f, out, m, n, x.stride(0), block)
    return out


def run_feature_bias(x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    block = triton.next_power_of_2(n)
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    feature_bias_2d[(m,)](x, f, out, m, n, x.stride(0), block)
    return out
