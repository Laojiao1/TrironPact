"""第五阶段显式复制修复：一次新分配、逻辑复制、完整二次复验。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pact.guard_plan import GuardPlan, GuardResult


@dataclass
class RepairResult:
    status: str
    tensors: dict[str, torch.Tensor]
    guard: GuardResult
    copied: tuple[str, ...]
    reason: str


def try_relayout(plan: GuardPlan, tensors: dict[str, torch.Tensor], rejected: GuardResult) -> RepairResult:
    current = dict(tensors)
    copied: list[str] = []
    guard = rejected
    names = {item.pointer: item.tensor for item in plan.spec.meaning.inputs} if plan.spec.meaning else {}
    for _ in range(len(names)):
        candidate = guard.failed
        if candidate is None or candidate.purpose not in ("Semantics", "Optimization"):
            break
        name = names.get(candidate.origin.pointer)
        if name is None or name in copied or name not in current:
            break
        source = current[name]
        if not isinstance(source, torch.Tensor) or source.device.type != "cuda" or source.dtype != torch.float32:
            break
        # contiguous() 可原样返回 offset 视图；这里强制新 storage。
        target = torch.empty(tuple(source.shape), dtype=source.dtype, device=source.device)
        target.copy_(source)
        current[name] = target
        copied.append(name)
        guard = plan.evaluate(current)
        if guard.allowed:
            return RepairResult("True", current, guard, tuple(copied), "复制后全部义务复验通过")
    return RepairResult("False" if guard.status == "False" else "Unknown", current, guard,
                        tuple(copied), "没有受支持的进一步修复或复验未通过")
