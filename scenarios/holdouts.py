"""第三阶段留出 Kernel 与独立语义；不提供目标布局谓词或 Guard。"""

import torch
import triton
import triton.language as tl

from pact.semantics import InputMeaning, OperatorMeaning


@triton.jit
def strided_vector(X, Y, N: tl.constexpr, S: tl.constexpr, B: tl.constexpr):
    idx = tl.program_id(0) * B + tl.arange(0, B)
    mask = idx < N
    value = tl.load(X + idx * S, mask=mask, other=0)
    tl.store(Y + idx, value + 1, mask=mask)


@triton.jit
def two_stride_matrix(X, Y, M: tl.constexpr, N: tl.constexpr, S0: tl.constexpr, S1: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    mask = col < N
    value = tl.load(X + row * S0 + col * S1, mask=mask, other=0)
    tl.store(Y + row * N + col, value + 1, mask=mask)


def run_strided_vector(x: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    n = x.shape[0]
    stride = x.stride(0)
    block = 128
    strided_vector[(triton.cdiv(n, block),)](x, out, n, stride, block)
    return out


def run_two_stride_matrix(x: torch.Tensor) -> torch.Tensor:
    m, n = x.shape
    out = torch.empty((m, n), dtype=x.dtype, device=x.device)
    row_stride = x.stride(0)
    col_stride = x.stride(1)
    block = triton.next_power_of_2(n)
    two_stride_matrix[(m,)](x, out, m, n, row_stride, col_stride, block)
    return out


# 外部逻辑规格只陈述 x 的逻辑索引和调用参数来源，不预填 stride 必须满足的目标谓词。
VECTOR_MEANING = OperatorMeaning(
    "holdout_vector", ("i",), "0 <= i < N",
    (InputMeaning("X", "X", ("N",), "x[i]"),),
    "Y", "out[i] = x[i] + 1",
    (("N", "x.size(0)"), ("S", "x.stride(0)"), ("B", "128"), ("grid", "ceil(N/B)")),
    "scenarios.cases.reference", "reference(x) = x + 1",
    ("CUDA float32", "N 至少为 1", "非负输入步长", "显式尾部 mask"),
)

MATRIX_MEANING = OperatorMeaning(
    "holdout_matrix", ("row", "col"), "0 <= row < M，0 <= col < N",
    (InputMeaning("X", "X", ("M", "N"), "x[row, col]"),),
    "Y", "out[row, col] = x[row, col] + 1",
    (("M", "x.size(0)"), ("N", "x.size(1)"), ("S0", "x.stride(0)"), ("S1", "x.stride(1)"), ("B", "next_power_of_2(N)"), ("grid", "M")),
    "scenarios.cases.reference", "reference(x) = x + 1",
    ("CUDA float32", "M、N 至少为 1", "非负输入步长", "显式尾部 mask"),
)
