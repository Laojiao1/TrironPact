"""第六阶段基准的独立规格、调用绑定和参考实现登记。"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from pact.candidates import CallBinding
from pact.semantics import InputMeaning, OperatorMeaning
from scenarios.bench_kernels import (
    run_axpy, run_feature_bias, run_feature_scale, run_multiply, run_relu, run_square,
)


@dataclass(frozen=True)
class AuditRecord:
    wrapper: str
    reference: str
    parameter_bindings: tuple[tuple[str, str], ...]
    dtype_domain: tuple[str, ...]
    input_domain: tuple[str, ...]
    core_contract: tuple[str, ...]
    allowed_paths: tuple[str, ...]
    exclusion_reason: str
    semantic_audit: str
    candidate_status: str = "complete"
    guard_status: str = "available"
    fast_status: str = "feasible"

    def to_dict(self) -> dict:
        return asdict(self)


def reference_relu(x: torch.Tensor) -> torch.Tensor:
    return torch.relu(x)


def reference_square(x: torch.Tensor) -> torch.Tensor:
    return x * x


def reference_multiply(x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return x * z


def reference_axpy(x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    return 2.0 * x + z


def reference_feature_scale(x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    return x * f


def reference_feature_bias(x: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    return x + f


NEW_CASES = {
    "bench_relu": (run_relu, reference_relu, "out[i] = max(x[i], 0)", (("X", "X"), ("Y", "OUT")), (("N", "x.size(0)"), ("B", "128")), "ceil(N/B)", ("N",)),
    "bench_square": (run_square, reference_square, "out[i] = x[i] * x[i]", (("X", "X"), ("Y", "OUT")), (("N", "x.size(0)"), ("B", "128")), "ceil(N/B)", ("N",)),
    "bench_multiply": (run_multiply, reference_multiply, "out[i] = x[i] * z[i]", (("X", "X"), ("Z", "Z"), ("Y", "OUT")), (("N", "x.size(0)"), ("B", "128")), "ceil(N/B)", ("N",)),
    "bench_axpy": (run_axpy, reference_axpy, "out[i] = 2*x[i] + z[i]", (("X", "X"), ("Z", "Z"), ("Y", "OUT")), (("ALPHA", "2"), ("N", "x.size(0)"), ("B", "128")), "ceil(N/B)", ("N",)),
    "bench_feature_scale": (run_feature_scale, reference_feature_scale, "out[row,col] = x[row,col] * f[col]", (("X", "X"), ("F", "F"), ("Y", "OUT")), (("M", "x.size(0)"), ("N", "x.size(1)"), ("S0", "x.stride(0)"), ("B", "next_power_of_2(N)")), "M", ("M", "N")),
    "bench_feature_bias": (run_feature_bias, reference_feature_bias, "out[row,col] = x[row,col] + f[col]", (("X", "X"), ("F", "F"), ("Y", "OUT")), (("M", "x.size(0)"), ("N", "x.size(1)"), ("S0", "x.stride(0)"), ("B", "next_power_of_2(N)")), "M", ("M", "N")),
}


def new_meaning(name: str) -> OperatorMeaning:
    _, reference, rule, pointers, scalars, grid, shape = NEW_CASES[name]
    feature = name.startswith("bench_feature_")
    inputs = (InputMeaning("X", "X", shape, "x[row, col]" if feature else "x[i]"),)
    if any(tensor == "Z" for _, tensor in pointers):
        inputs += (InputMeaning("Z", "Z", shape, "z[i]"),)
    if feature:
        inputs += (InputMeaning("F", "F", ("N",), "f[col]"),)
    axes = ("row", "col") if feature else ("i",)
    domain = "0 <= row < M，0 <= col < N" if feature else "0 <= i < N"
    return OperatorMeaning(name, axes, domain, inputs, "Y", rule, scalars + (("grid", grid),),
                           f"bench.specs.{reference.__name__}", rule,
                           ("CUDA float32", "各维长度至少为 1", "显式尾部 mask", "输出连续新分配"))


def new_call(name: str) -> CallBinding:
    _, _, _, pointers, scalars, grid, shape = NEW_CASES[name]
    return CallBinding(pointers, scalars, grid, shape, 4, f"scenarios.bench_kernels.{NEW_CASES[name][0].__name__}")


def audit_records() -> dict[str, AuditRecord]:
    records = {
        "A": AuditRecord("scenarios.kernels.run_fast[A]", "scenarios.cases.reference", (("M", "x.size(0)"), ("N", "x.size(1)"), ("B", "128"), ("grid", "ceil(M*N/B)")), ("torch.float32",), ("二维非空", "元素数不超过 1000000"), ("stride(0)==size(1) or size(0)==1", "stride(1)==1 or size(1)==1", "span within storage"), ("Fast", "Relayout+Fast", "PyTorch Fallback"), "", "pact.semantics.SEMANTICS[A] and phase3 candidate audit"),
        "A2": AuditRecord("scenarios.kernels.run_fast[A2]", "scenarios.cases.reference", (("M", "x.size(0)"), ("N", "x.size(1)"), ("S0", "x.stride(0)"), ("B", "next_power_of_2(N)"), ("grid", "M")), ("torch.float32",), ("二维非空", "N<=1024"), ("stride(1)==1 or size(1)==1", "span within storage"), ("Fast", "Relayout+Fast", "PyTorch Fallback"), "", "pact.semantics.SEMANTICS[A2] and phase3 candidate audit"),
        "B": AuditRecord("scenarios.kernels.run_fast[B]", "scenarios.cases.reference", (("N", "x.size(0)"), ("B", "128"), ("grid", "ceil(N/B)")), ("torch.float32",), ("一维非空", "N<=1000000"), ("stride(0)==1 or size(0)==1", "span within storage"), ("Fast", "Relayout+Fast", "PyTorch Fallback"), "", "pact.semantics.SEMANTICS[B] and phase3 candidate audit"),
        "D": AuditRecord("scenarios.external_add.run_add", "scenarios.cases.reference", (("n_elements", "x.numel()=y.numel()"), ("BLOCK_SIZE", "128"), ("grid", "ceil(n_elements/BLOCK_SIZE)")), ("torch.float32",), ("两个等长一维 CUDA 输入",), ("X/Y stride(0)==1 or size(0)==1", "span within storage"), ("Fast", "Relayout+Fast", "PyTorch Fallback"), "", "pact.semantics.SEMANTICS[D] and pinned Triton source"),
        "holdout_vector": AuditRecord("scenarios.holdouts.run_strided_vector", "scenarios.cases.reference", (("N", "x.size(0)"), ("S", "x.stride(0)"), ("B", "128"), ("grid", "ceil(N/B)")), ("torch.float32",), ("一维非空", "非负步长"), ("stride(0)==S", "span within storage"), ("Fast", "PyTorch Fallback"), "", "scenarios.holdouts.VECTOR_MEANING and phase3 holdout audit"),
        "holdout_matrix": AuditRecord("scenarios.holdouts.run_two_stride_matrix", "scenarios.cases.reference", (("M", "x.size(0)"), ("N", "x.size(1)"), ("S0", "x.stride(0)"), ("S1", "x.stride(1)"), ("B", "next_power_of_2(N)"), ("grid", "M")), ("torch.float32",), ("二维非空", "非负步长"), ("stride(0)==S0", "stride(1)==S1", "span within storage"), ("Fast", "PyTorch Fallback"), "", "scenarios.holdouts.MATRIX_MEANING and phase3 holdout audit"),
        "pointer_hint": AuditRecord("scenarios.alignment_fixtures.run_pointer_hint", "scenarios.cases.reference", (("N", "x.size(0)"), ("B", "128"), ("grid", "ceil(N/B)")), ("torch.float32",), ("一维非空",), ("effective_ptr%16==0", "stride(0)==1 or size(0)==1", "span within storage"), ("Fast", "PyTorch Fallback"), "", "phase3 alignment fixture audit"),
        "index_hint": AuditRecord("scenarios.alignment_fixtures.run_index_hint", "scenarios.cases.reference", (("N", "x.size(0)"), ("B", "128"), ("grid", "ceil(N/B)")), ("torch.float32",), ("一维非空",), ("index multiple_of(16)", "stride(0)==1 or size(0)==1", "span within storage"), ("Fast", "PyTorch Fallback"), "", "phase3 alignment fixture audit"),
        "vllm_expand_idx": AuditRecord("not_applicable_challenge", "vLLM expand_idx_mapping semantics", (), ("upstream declared dtype domain",), ("固定上游源码；未执行",), (), ("Unsupported",), "无显式 mask 的标量 load、多个 store 和间接区间，不在当前支持域", "pinned upstream source review", "unsupported", "unsupported", "unsupported"),
    }
    common_domain = ("CUDA float32", "非空张量", "显式 mask", "受限单轴 grid")
    for name, (wrapper, reference, _, _, scalars, _, _) in NEW_CASES.items():
        feature = name.startswith("bench_feature_")
        records[name] = AuditRecord(
            f"scenarios.bench_kernels.{wrapper.__name__}", f"bench.specs.{reference.__name__}", scalars,
            ("torch.float32",), common_domain + (("X 为二维且 F 长度等于 X.size(1)",) if feature else ("输入为等长一维张量",)),
            (("X.stride(1)==1 or X.size(1)==1", "F.stride(0)==1 or F.size(0)==1") if feature else ("所有输入 stride(0)==1 或 size(0)==1",)),
            ("Offline Shadow",), "未注册 GuardPlan，不允许在线 Fast", "independent_spec_and_wrapper_reviewed",
            "shadow_complete", "not_registered", "not_eligible")
    return records


AUDIT_RECORDS = audit_records()
