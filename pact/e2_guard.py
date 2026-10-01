"""e2 冻结调用域的保守 GuardPlan；尚不接入在线分派。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import torch

from pact.candidates import Candidate
from pact.contract_dsl import TensorMetadata
from pact.e2_contracts import FrozenTensor


@dataclass(frozen=True)
class E2GuardResult:
    status: str
    reason: str
    trace: tuple[dict, ...]

    @property
    def allowed(self) -> bool:
        return self.status == "True"


@dataclass(frozen=True)
class E2GuardPlan:
    """只在冻结 shape/stride/dtype/grid 域内放行，不声明一般布局安全。"""

    kernel_id: str
    expected: tuple[FrozenTensor, ...]
    candidates: tuple[Candidate, ...]
    status: str
    reason: str
    fingerprint: str

    def evaluate(self, tensors: dict[str, torch.Tensor]) -> E2GuardResult:
        trace: list[dict] = []

        def stop(status: str, reason: str) -> E2GuardResult:
            return E2GuardResult(status, reason, tuple(trace))

        if self.status != "Supported":
            return stop(self.status, self.reason)
        expected = {item.name: item for item in self.expected}
        if set(tensors) != set(expected):
            return stop("Unknown", "Tensor 绑定集合与冻结规格不一致")
        for name, spec in expected.items():
            tensor = tensors[name]
            if not isinstance(tensor, torch.Tensor):
                return stop("Unknown", f"{name} 不是 Tensor")
            if tensor.device.type != "cuda":
                return stop("False", f"{name} 不在 CUDA")
            if str(tensor.dtype).removeprefix("torch.") != spec.dtype:
                return stop("False", f"{name} dtype 与冻结规格不一致")
            if tuple(tensor.shape) != spec.shape or tuple(tensor.stride()) != spec.stride:
                return stop("False", f"{name} shape/stride 与冻结规格不一致")
            if tensor.storage_offset() != 0 or tensor.element_size() != spec.element_bytes:
                return stop("False", f"{name} storage_offset/element_bytes 与冻结规格不一致")
        trace.append({"stage": "metadata", "value": True})
        storages = [tensor.untyped_storage().data_ptr() for tensor in tensors.values()]
        if len(storages) != len(set(storages)):
            return stop("Unknown", "冻结域禁止输入输出共享 storage")
        trace.append({"stage": "alias", "value": True})
        metadata = {
            name: TensorMetadata(
                tuple(tensor.shape),
                tuple(tensor.stride()),
                tensor.storage_offset(),
                tensor.data_ptr(),
                tensor.element_size(),
                tensor.untyped_storage().nbytes(),
            )
            for name, tensor in tensors.items()
        }
        for candidate in self.candidates:
            value = candidate.evaluate(metadata)
            trace.append(
                {
                    "stage": "span" if candidate.condition and candidate.condition.kind == "access_span" else "metadata",
                    "value": value,
                    "rule": candidate.rule,
                    "pointer": candidate.origin.pointer,
                    "line": candidate.origin.source_line,
                }
            )
            if value is not True:
                return stop("False" if value is False else "Unknown", "候选义务未满足")
        return stop("True", "冻结调用域的全部访存义务本次为真")

    def to_dict(self) -> dict:
        return {
            "kernel_id": self.kernel_id,
            "expected": [asdict(item) for item in self.expected],
            "candidates": [item.to_dict() for item in self.candidates],
            "status": self.status,
            "reason": self.reason,
            "fingerprint": self.fingerprint,
        }


def compile_e2_guard(kernel_id: str, expected: tuple[FrozenTensor, ...], candidates: tuple[Candidate, ...], source_fingerprint: str) -> E2GuardPlan:
    reasons: list[str] = []
    if not kernel_id or not expected:
        reasons.append("缺少 Kernel ID 或冻结 Tensor 规格")
    if not candidates:
        reasons.append("缺少候选")
    access_origins = {(item.origin.source_file, item.origin.source_line, item.origin.pointer) for item in candidates}
    for candidate in candidates:
        if candidate.condition is None or candidate.evidence != "Statically-Proven":
            reasons.append("候选缺少可执行条件或静态访存证据")
        if not candidate.binding or not candidate.domain or not candidate.origin:
            reasons.append("候选缺少绑定、适用域或访问点来源")
    if any(item.element_bytes < 1 or not item.shape or len(item.shape) != len(item.stride) for item in expected):
        reasons.append("冻结 Tensor 元数据无效")
    payload = json.dumps(
        {
            "kernel_id": kernel_id,
            "expected": [asdict(item) for item in expected],
            "candidates": [item.to_dict() for item in candidates],
            "source_fingerprint": source_fingerprint,
            "origins": sorted(access_origins),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    fingerprint = hashlib.sha256(payload.encode()).hexdigest()
    return E2GuardPlan(
        kernel_id,
        expected,
        candidates,
        "Unknown" if reasons else "Supported",
        "; ".join(dict.fromkeys(reasons)) if reasons else "冻结调用域的候选、来源和元数据条件完整",
        fingerprint,
    )
