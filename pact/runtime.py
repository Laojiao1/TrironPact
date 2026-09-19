"""运行时 Guard 检查（显存跨度、dtype、stride 谓词求值）与 Fast/Fallback 分派"""

from dataclasses import dataclass
from functools import lru_cache

import torch

from pact.analysis import Analysis, Predicate, extract
from pact.refine import apply_proven_singleton_rule
from scenarios.kernels import run_fast


@dataclass(frozen=True)
class GuardResult:
    allowed: bool
    reason: str
    evidence: str
    checks: tuple[str, ...]
    proof_scope: str = "仅声明语义与受支持模板内的访存布局兼容性"


@lru_cache(maxsize=None)
def generated_contract(name: str, refined: bool = False) -> Analysis:
    """每个 Kernel 只解析一次；谓词来自其 AST 提取结果。"""
    initial = extract(name)
    return apply_proven_singleton_rule(initial) if refined else initial


# 谓词求值：将提取出的符号约束（如 size(1)）结合输入张量的运行时实际属性进行计算比对
def _predicate_holds(predicate: Predicate, x: torch.Tensor, y: torch.Tensor | None = None) -> bool | None:
    target = y if predicate.tensor == "Y" else x
    if target is None:
        return None
    if predicate.unless_size_one is not None and target.size(predicate.unless_size_one) == 1:
        return True
    if predicate.expected == "1":
        expected = 1
    elif predicate.expected == "size(1)" and target.ndim == 2:
        expected = target.size(1)
    else:
        return None
    return target.stride(predicate.dimension) == expected


# 跨度安全检查：通过张量的 shape 和 stride 计算该视图访问的最大物理元素索引，确保整个寻址跨度严格落在该张量的真实底层显存（untyped_storage）内，严防 GPU 越界访存
def _span_within_storage(x: torch.Tensor) -> bool:
    """offset 是相对 storage 的元素数；这里不再从 data_ptr 加一次。"""
    if x.numel() == 0 or any(step < 0 for step in x.stride()):
        return False
    last = x.storage_offset() + sum((size - 1) * step for size, step in zip(x.shape, x.stride()))
    return last >= x.storage_offset() >= 0 and (last + 1) * x.element_size() <= x.untyped_storage().nbytes()


def guard(name: str, x: torch.Tensor, y: torch.Tensor | None = None, *, refined: bool = False) -> GuardResult:
    if name not in ("A", "A2", "B", "D"):
        return GuardResult(False, "未知 Kernel", "Unsupported", ())
    contract = generated_contract(name, refined)
    if contract.status != "Supported":
        return GuardResult(False, contract.reason, contract.status, ())
    if not isinstance(x, torch.Tensor):
        return GuardResult(False, "输入不是 Tensor", "Unknown", ())
    if x.device.type != "cuda" or x.dtype != torch.float32 or x.ndim != contract.semantics.ndim:
        return GuardResult(False, "设备、类型或维度不在支持域", "Unknown", ())
    if name == "D" and (not isinstance(y, torch.Tensor) or y.device != x.device or y.dtype != x.dtype or y.shape != x.shape):
        return GuardResult(False, "第二个输入的设备、类型或形状不匹配", "Unknown", ())
    if any(size < 1 for size in x.shape) or x.numel() > 1_000_000:
        return GuardResult(False, "尺寸不在支持域", "Unknown", ())
    if name == "A2" and (x.size(0) < 2 or x.size(1) < 2):
        return GuardResult(False, "二维边界尺寸暂不分析", "Unknown", ())
    # A2 先限于不重叠的正常行视图，避免把更宽的输入域误称为已证明。
    if name == "A2" and (x.size(1) > 1024 or x.stride(0) < x.size(1)):
        return GuardResult(False, "行跨度不在已分析的非重叠子域", "Unknown", ())
    if not _span_within_storage(x):
        return GuardResult(False, "访问范围不能落入已知 storage", "Unknown", ())
    if name == "D" and not _span_within_storage(y):
        return GuardResult(False, "第二个输入的访问范围不能落入 storage", "Unknown", ())

    checks: list[str] = []
    # 逐条执行从 AST 推出的谓词，而不是复制一份手写布局规则。
    for predicate in contract.predicates:
        result = _predicate_holds(predicate, x, y)
        condition = f"{predicate.tensor}.stride({predicate.dimension}) == {predicate.expected}"
        if predicate.unless_size_one is not None:
            condition += f" 或 size({predicate.unless_size_one}) == 1"
        checks.append(f"{condition}: {result}")
        if result is None:
            return GuardResult(False, "生成的谓词无法求值", "Unknown", tuple(checks))
        if not result:
            return GuardResult(False, "输入布局不满足提取条件", "Statically-Proven", tuple(checks))
    return GuardResult(True, "在声明语义和有限模板内满足静态条件", "Statically-Proven", tuple(checks))


def dispatch(name: str, x: torch.Tensor, y: torch.Tensor | None = None, *, refined: bool = False) -> tuple[torch.Tensor, str, GuardResult]:
    if name == "D" and y is None:
        raise ValueError("向量加法需要第二个输入")
    decision = guard(name, x, y, refined=refined)
    if decision.allowed:
        return run_fast(name, x, y), "Fast", decision
    return (x + y if name == "D" and y is not None else x + 1), "PyTorch Fallback", decision
