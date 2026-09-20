"""第五阶段：从当前 Access IR、独立语义和已审调用绑定编译 Guard。"""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import triton

from pact.access_ir import parse_access_ir
from pact.alignment import extract_alignment
from pact.call_site import audit_call_site
from pact.candidates import CallBinding, Candidate
from pact.contract_dsl import Condition, TensorMetadata
from pact.semantics import SEMANTICS, OperatorMeaning
from pact.shape_stride import extract_shape_stride
from pact.span import extract_spans
from scenarios.alignment_fixtures import index_hint, pointer_hint, run_index_hint, run_pointer_hint
from scenarios.external_add import add_kernel, run_add
from scenarios.holdouts import MATRIX_MEANING, VECTOR_MEANING, run_strided_vector, run_two_stride_matrix, strided_vector, two_stride_matrix
from scenarios.kernels import KERNELS, run_fast


RULE_VERSION = "guard-plan-1"


@dataclass(frozen=True)
class KernelSpec:
    kernel: object
    wrapper: object
    meaning: OperatorMeaning


SPECS = {
    "A": KernelSpec(KERNELS["A"], run_fast, SEMANTICS["A"]),
    "A2": KernelSpec(KERNELS["A2"], run_fast, SEMANTICS["A2"]),
    "B": KernelSpec(KERNELS["B"], run_fast, SEMANTICS["B"]),
    "D": KernelSpec(add_kernel, run_add, SEMANTICS["D"]),
    "holdout_vector": KernelSpec(strided_vector, run_strided_vector, VECTOR_MEANING),
    "holdout_matrix": KernelSpec(two_stride_matrix, run_two_stride_matrix, MATRIX_MEANING),
    "pointer_hint": KernelSpec(pointer_hint, run_pointer_hint, SEMANTICS["B"]),
    "index_hint": KernelSpec(index_hint, run_index_hint, SEMANTICS["B"]),
}


@dataclass(frozen=True)
class Check:
    stage: str
    candidate: Candidate

    def to_dict(self) -> dict:
        return {"stage": self.stage, "candidate": self.candidate.to_dict()}


@dataclass
class GuardResult:
    status: str
    reason: str
    checks: list[dict]
    output: torch.Tensor | None = None
    failed: Candidate | None = None
    scalars: dict[str, int] | None = None

    @property
    def allowed(self) -> bool:
        return self.status == "True"

    def to_dict(self) -> dict:
        return {"status": self.status, "reason": self.reason, "checks": self.checks,
                "failed": self.failed.to_dict() if self.failed else None,
                "output": tuple(self.output.shape) if self.output is not None else None,
                "scalars": self.scalars}


def _stage(condition: Condition) -> str:
    if condition.kind == "access_span" or condition.kind == "span_in_storage":
        return "span"
    if condition.kind in ("and", "or"):
        return max((_stage(part) for part in condition.parts), key={"metadata": 0, "pointer": 1, "span": 2}.__getitem__)
    if condition.left is not None and condition.left.kind == "effective_ptr":
        return "pointer"
    if condition.right is not None and condition.right.kind == "effective_ptr":
        return "pointer"
    return "metadata"


def _metadata(t: torch.Tensor, *, physical: bool) -> TensorMetadata:
    return TensorMetadata(tuple(t.shape), tuple(t.stride()), t.storage_offset(),
                          t.data_ptr() if physical else 0, t.element_size(),
                          t.untyped_storage().nbytes() if physical else 0)


def _binding_value(expr: str, tensors: dict[str, torch.Tensor], scalars: dict[str, int]) -> int | None:
    if expr.isdigit():
        return int(expr)
    if expr in scalars:
        return scalars[expr]
    match = re.fullmatch(r"([xy])\.(size|stride)\(([0-9]+)\)", expr)
    if match:
        t = tensors.get(match[1].upper())
        axis = int(match[3])
        if t is None or axis >= t.ndim:
            return None
        return int(t.size(axis) if match[2] == "size" else t.stride(axis))
    match = re.fullmatch(r"([xy])\.numel\(\)", expr)
    if match:
        t = tensors.get(match[1].upper())
        return int(t.numel()) if t is not None else None
    match = re.fullmatch(r"next_power_of_2\(([A-Za-z][A-Za-z0-9_]*)\)", expr)
    if match and match[1] in scalars:
        return triton.next_power_of_2(scalars[match[1]])
    if expr == "x.numel() = y.numel()":
        x, y = tensors.get("X"), tensors.get("Y")
        return int(x.numel()) if x is not None and y is not None and x.numel() == y.numel() else None
    return None


def _grid(expr: str, scalars: dict[str, int]) -> int | None:
    if expr in scalars:
        return scalars[expr]
    match = re.fullmatch(r"ceil\(([A-Za-z_*]+)\/([A-Za-z_]+)\)", expr)
    if match:
        numerators = match[1].split("*")
        if any(item not in scalars for item in numerators) or match[2] not in scalars:
            return None
        numerator = 1
        for item in numerators:
            numerator *= scalars[item]
        denominator = scalars[match[2]]
        return triton.cdiv(numerator, denominator) if denominator > 0 else None
    return None


@dataclass(frozen=True)
class GuardPlan:
    name: str
    spec: KernelSpec
    binding: CallBinding | None
    signature: tuple[str, ...]
    checks: tuple[Check, ...]
    fingerprint: str
    status: str
    reason: str
    source_tokens: tuple[object, ...] = ()

    def evaluate(self, tensors: dict[str, torch.Tensor], *, output: torch.Tensor | None = None) -> GuardResult:
        trace: list[dict] = []
        def stop(status: str, reason: str, failed: Candidate | None = None) -> GuardResult:
            return GuardResult(status, reason, trace, output, failed)
        if self.status != "Supported" or self.binding is None:
            return stop(self.status, self.reason)
        names = tuple(item.tensor for item in self.spec.meaning.inputs)
        if set(tensors) != set(names):
            return stop("Unknown", "输入绑定集合不匹配")
        for name in names:
            value = tensors[name]
            if not isinstance(value, torch.Tensor):
                return stop("Unknown", f"{name} 不是 Tensor")
            if value.device.type != "cuda" or value.dtype != torch.float32:
                return stop("False", f"{name} 不是 CUDA float32")
        trace.append({"stage": "device_dtype", "value": True})
        shape = tuple(tensors[names[0]].shape)
        if len(shape) != self.spec.meaning.ndim or any(size < 1 for size in shape) or tensors[names[0]].numel() > 1_000_000:
            return stop("False", "rank、正尺寸或元素总数超出支持域")
        if any(tuple(tensors[name].shape) != shape or tensors[name].device != tensors[names[0]].device for name in names):
            return stop("False", "输入形状或设备不一致")
        if any(any(step < 0 for step in tensors[name].stride()) for name in names):
            return stop("Unknown", "负步长不在支持域")
        assumptions = self.spec.meaning.assumptions
        if ("M、N 至少为 2" in assumptions and min(shape) < 2) or ("N 不超过 1024" in assumptions and shape[-1] > 1024):
            return stop("False", "独立语义声明的尺寸域不满足")
        trace.append({"stage": "rank_shape", "value": True, "shape": shape})
        scalars: dict[str, int] = {}
        for key, expr in self.binding.scalars:
            value = _binding_value(expr, tensors, scalars)
            if value is None or type(value) is not int or value < 0:
                return stop("Unknown", f"标量绑定无法求值：{key}")
            scalars[key] = value
        if _grid(self.binding.grid, scalars) is None:
            return stop("Unknown", "grid 绑定无法求值")
        metadata = {name: _metadata(tensors[name], physical=False) for name in names}
        for check in self.checks:
            if check.stage != "metadata":
                continue
            value = check.candidate.evaluate(metadata, scalars)
            trace.append({"stage": "stride", "value": value, "rule": check.candidate.rule,
                          "pointer": check.candidate.origin.pointer, "purpose": check.candidate.purpose})
            if value is not True:
                return stop("False" if value is False else "Unknown", "shape/stride 或索引条件未满足", check.candidate)
        if "行跨度至少为 N" in assumptions and tensors["X"].stride(0) < shape[-1]:
            trace.append({"stage": "stride", "value": False, "rule": "independent_row_span_domain", "pointer": "X", "purpose": "Safety"})
            return stop("False", "独立语义声明的行跨度域不满足")
        for check in self.checks:
            if check.stage != "pointer":
                continue
            # 仅在这一层读取实际有效地址；storage_offset 不再加进 data_ptr。
            for name in names:
                metadata[name] = _metadata(tensors[name], physical=True)
            value = check.candidate.evaluate(metadata, scalars)
            trace.append({"stage": "pointer", "value": value, "rule": check.candidate.rule,
                          "pointer": check.candidate.origin.pointer, "purpose": check.candidate.purpose})
            if value is not True:
                return stop("False" if value is False else "Unknown", "实际有效地址对齐义务未满足", check.candidate)
        if output is None:
            output = torch.empty(shape, dtype=tensors[names[0]].dtype, device=tensors[names[0]].device)
        if not isinstance(output, torch.Tensor) or tuple(output.shape) != shape or output.dtype != torch.float32 or output.device != tensors[names[0]].device or not output.is_contiguous():
            return stop("Unknown", "实际输出分配不符合已审形状、dtype、设备或布局")
        if any(output.untyped_storage().data_ptr() == tensors[name].untyped_storage().data_ptr() for name in names):
            return stop("Unknown", "输出与输入共享 storage")
        metadata = {name: _metadata(tensors[name], physical=True) for name in names}
        metadata["OUT"] = _metadata(output, physical=True)
        for check in self.checks:
            if check.stage != "span":
                continue
            value = check.candidate.evaluate(metadata, scalars)
            trace.append({"stage": "span", "value": value, "rule": check.candidate.rule,
                          "pointer": check.candidate.origin.pointer, "purpose": check.candidate.purpose})
            if value is not True:
                return stop("False" if value is False else "Unknown", "实际输入或输出访问 span 未满足", check.candidate)
        result = stop("True", "全部必要义务本次为真")
        result.scalars = scalars
        return result

    def launch(self, tensors: dict[str, torch.Tensor], result: GuardResult) -> torch.Tensor:
        if not result.allowed or result.output is None or result.scalars is None or self.binding is None:
            raise ValueError("Fast 启动缺少通过的 Guard 与实际输出")
        pointer_map = dict(self.binding.pointers)
        args = [result.output if pointer_map.get(param) == "OUT" else tensors[pointer_map[param]]
                if param in pointer_map else result.scalars[param]
                for param in self.signature]
        grid = _grid(self.binding.grid, result.scalars)
        self.spec.kernel[(grid,)](*args)
        return result.output

    def to_dict(self) -> dict:
        return {"version": RULE_VERSION, "name": self.name, "fingerprint": self.fingerprint,
                "status": self.status, "reason": self.reason,
                "binding": self.binding.to_dict() if self.binding else None, "signature": self.signature,
                "checks": [item.to_dict() for item in self.checks]}


def compile_guard_plan(name: str, *, spec: KernelSpec | None = None) -> GuardPlan:
    spec = spec or SPECS.get(name)
    if spec is None:
        return GuardPlan(name, KernelSpec(None, None, None), None, (), (), "", "Unsupported", "未知 Kernel")
    meaning = spec.meaning
    pointers = {item.pointer: item.tensor for item in meaning.inputs}
    pointers[meaning.output_pointer] = "OUT"
    ir = parse_access_ir(spec.kernel, pointers, {key: 4 for key in pointers})
    declared = dict(meaning.launch_bindings)
    binding = CallBinding(tuple(pointers.items()), tuple((k, v) for k, v in meaning.launch_bindings if k != "grid"),
                          declared.get("grid", ""), meaning.inputs[0].shape_params, 4, "当前已核对 wrapper")
    try:
        site = audit_call_site(spec.wrapper, spec.kernel.fn.__name__, binding)
        source = inspect.getsource(spec.kernel.fn) + inspect.getsource(spec.wrapper)
    except (AttributeError, OSError, TypeError) as error:
        return GuardPlan(name, spec, binding, ir.signature, (), "", "Unknown", f"源码不可核对：{error}")
    groups = (extract_shape_stride(ir, meaning, binding, site), extract_spans(ir, meaning, binding, site),
              extract_alignment(ir, meaning, binding, site))
    candidates = tuple(item for group in groups for item in group.candidates)
    payload = json.dumps({"version": RULE_VERSION, "guard_implementation": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          "source": source, "meaning": asdict(meaning),
                          "ir": ir.to_dict(), "binding": binding.to_dict(),
                          "candidates": [item.to_dict() for item in candidates]}, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(payload.encode()).hexdigest()
    reasons = [f"{ir.status}: {ir.reason}" if ir.status != "Supported" else ""]
    reasons += list(site.reasons)
    reasons += [f"{group.status}: {group.reason}" for group in groups if group.status != "Supported"]
    reasons += ["候选缺失条件、来源、绑定或充分证据" for item in candidates if item.condition is None or item.evidence not in ("Statically-Proven", "Unproven") or not item.domain or not item.rule or not item.binding or not item.origin]
    valid_origins = {(point.source_file, point.line, point.pointer_param) for point in ir.accesses}
    for item in candidates:
        extra_domain = item.domain[len(meaning.assumptions):] if item.domain[:len(meaning.assumptions)] == meaning.assumptions else None
        if item.binding != meaning.launch_bindings or extra_domain not in ((), ("形状正长度", "行系数非负")):
            reasons.append("候选的绑定或适用域与独立语义冲突")
        if (item.origin.source_file, item.origin.source_line, item.origin.pointer) not in valid_origins:
            reasons.append("候选没有当前访存点来源")
        if item.evidence == "Unproven" and not (item.purpose == "Optimization" and item.rule == "multiple_of_pointer_base" and item.condition is not None and item.condition.kind == "mod_eq" and item.condition.left.kind == "effective_ptr"):
            reasons.append("Unproven 义务不在可在线消解的受限指针提示规则内")
    for point in ir.accesses:
        if not any(item.purpose == "Safety" and item.origin.source_line == point.line and item.origin.pointer == point.pointer_param for item in candidates):
            reasons.append(f"缺失访存 Safety 义务：{point.pointer_param}:{point.line}")
        if point.kind == "load" and not any(item.purpose == "Semantics" and item.origin.source_line == point.line and item.origin.pointer == point.pointer_param for item in candidates):
            reasons.append(f"缺失 load Semantics 义务：{point.pointer_param}:{point.line}")
    if len(ir.hints) != len(groups[2].candidates):
        reasons.append("提示义务数量不完整")
    for item in groups[2].candidates:
        if item.evidence == "Unproven" and not (item.purpose == "Optimization" and item.rule == "multiple_of_pointer_base" and item.condition is not None and item.condition.kind == "mod_eq" and item.condition.left.kind == "effective_ptr"):
            reasons.append("Unproven 优化义务不能在线消解")
    checks = tuple(sorted((Check(_stage(item.condition), item) for item in candidates if item.condition is not None),
                          key=lambda check: {"metadata": 0, "pointer": 1, "span": 2}[check.stage]))
    reason = "; ".join(part for part in reasons if part)
    return GuardPlan(name, spec, binding, ir.signature, checks, digest, "Unknown" if reason else "Supported", reason or "当前源码、语义和全部访存义务已核对",
                     (spec.kernel.fn.__code__, spec.wrapper.__code__))
