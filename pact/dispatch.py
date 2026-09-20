"""第五阶段独立在线入口；旧 pact.runtime 的 PoC 行为保持原样。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from pact.cost_model import CostTable
from pact.guard_plan import GuardPlan, GuardResult, SPECS, compile_guard_plan
from pact.relayout import RepairResult, try_relayout
from scenarios.cases import reference


_PLANS: dict[str, GuardPlan] = {}


def get_plan(name: str) -> GuardPlan:
    current = SPECS.get(name)
    cached = _PLANS.get(name)
    # 活进程中函数对象变更会失效；离线序列化指纹由编译时来源核对。
    if cached is None or cached.spec is not current or (current is not None and cached.source_tokens != (current.kernel.fn.__code__, current.wrapper.__code__)):
        cached = compile_guard_plan(name)
        _PLANS[name] = cached
    return cached


def _fallback_defined(tensors: dict[str, torch.Tensor], plan: GuardPlan) -> bool:
    meaning = plan.spec.meaning
    if meaning is None or set(tensors) != {item.tensor for item in meaning.inputs}:
        return False
    if any(not isinstance(t, torch.Tensor) or t.dtype not in (torch.float16, torch.float32, torch.float64) for t in tensors.values()):
        return False
    values = list(tensors.values())
    return all(t.device == values[0].device and tuple(t.shape) == tuple(values[0].shape) for t in values)


@dataclass
class DispatchResult:
    path: str
    output: torch.Tensor | None
    direct_guard: GuardResult
    repair: RepairResult | None
    feasible: tuple[str, ...]
    selection_reason: str
    fingerprint: str

    def to_dict(self) -> dict:
        return {"path": self.path, "direct_guard": self.direct_guard.to_dict(),
                "repair": {"status": self.repair.status, "copied": self.repair.copied,
                           "reason": self.repair.reason, "guard": self.repair.guard.to_dict()} if self.repair else None,
                "feasible": self.feasible, "selection_reason": self.selection_reason,
                "fingerprint": self.fingerprint}


def dispatch(name: str, x: torch.Tensor, y: torch.Tensor | None = None, *,
             cost_table: CostTable | None = None, repair_policy: str = "cost") -> DispatchResult:
    """repair_policy='prefer_repair' 仅供隔离路径验收，仍执行完整复验。"""
    if repair_policy not in ("cost", "prefer_repair"):
        raise ValueError("未知修复策略")
    plan = get_plan(name)
    tensors = {"X": x}
    if y is not None:
        tensors["Y"] = y
    direct = plan.evaluate(tensors)
    feasible = set()
    if direct.allowed:
        feasible.add("Fast")
    fallback = _fallback_defined(tensors, plan)
    if fallback:
        feasible.add("PyTorch Fallback")
    repair = None
    if direct.status == "False" and direct.failed is not None and repair_policy == "prefer_repair":
        repair = try_relayout(plan, tensors, direct)
        if repair.status == "True":
            feasible.add("Relayout+Fast")
    if repair_policy == "prefer_repair" and "Relayout+Fast" in feasible:
        path, reason = "Relayout+Fast", "隔离验收显式选择已复验的修复路径"
    else:
        # 在线默认缺表不预先复制。只有显式离线表可启用经过复验的修复。
        if cost_table is not None and direct.status == "False" and direct.failed is not None and repair is None and cost_table.has_repair_entry(plan, tensors):
            repair = try_relayout(plan, tensors, direct)
            if repair.status == "True":
                feasible.add("Relayout+Fast")
        path, reason = (cost_table or CostTable()).select(plan, tensors, feasible, repair.copied if repair and repair.status == "True" else ())
    if path == "Fast":
        out = plan.launch(tensors, direct)
    elif path == "Relayout+Fast":
        out = plan.launch(repair.tensors, repair.guard)
    elif path == "PyTorch Fallback":
        out = reference(x, y)
    else:
        out = None
    return DispatchResult(path, out, direct, repair, tuple(sorted(feasible)), reason, plan.fingerprint)
