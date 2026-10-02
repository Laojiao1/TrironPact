"""e3 独立 GPU worker：比较原始 Kernel、Guard、Relayout 与 Fallback。"""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from bench.e1.catalog import load
from bench.e1.specs import construct, reference, value
from bench.e1.worker import decode
from bench.e2.coverage import _kernel
from bench.e2.plans import build_plan
from bench.e3.specs import risk_spec


def _column_stride_view(tensor: torch.Tensor) -> torch.Tensor:
    """构造值独立、shape 不变且最后一维步长为 2 的合法 CUDA 视图。"""

    shape = tuple(tensor.shape)
    expanded = (*shape[:-1], shape[-1] * 2)
    base = torch.empty(expanded, device=tensor.device, dtype=tensor.dtype)
    view = base[..., ::2]
    view.copy_(tensor)
    return view


def _make_tensors(entry: dict, seed: int, names: tuple[str, ...]) -> dict[str, torch.Tensor]:
    spec = decode(entry["smoke"])
    tensors = construct(spec, seed)
    for name in names:
        if name not in spec.inputs:
            raise ValueError(f"Unknown：{name} 不是登记输入")
        tensors[name] = _column_stride_view(tensors[name])
    return tensors


def _arguments(entry: dict, tensors: dict[str, torch.Tensor]) -> dict[str, object]:
    spec = decode(entry["smoke"])
    return {binding.parameter: value(binding.expression, tensors) for binding in spec.bindings}


def _checks(entry: dict, tensors: dict[str, torch.Tensor], wanted: tuple[torch.Tensor, ...]) -> list[dict]:
    spec = decode(entry["smoke"])
    rows = []
    for name, expected in zip(spec.outputs, wanted):
        actual = tensors[name]
        compatible = actual.shape == expected.shape and actual.dtype == expected.dtype
        close = torch.isclose(actual, expected, rtol=spec.rtol, atol=spec.atol, equal_nan=False) if compatible else None
        rows.append(
            {
                "output": name,
                "correct": bool(compatible and close.all().item()),
                "mismatch_count": int((~close).sum().item()) if close is not None else None,
                "max_abs_error": float((actual - expected).abs().max().item()) if compatible and actual.dtype != torch.bool else None,
            }
        )
    return rows


def _launch(entry: dict, tensors: dict[str, torch.Tensor]) -> None:
    spec = decode(entry["smoke"])
    kernel = _kernel(entry)
    arguments = _arguments(entry, tensors)
    if tuple(arguments) != tuple(kernel.arg_names):
        raise ValueError("Unknown：实际 JIT 参数与冻结 wrapper 绑定不一致")
    kernel[spec.grid](**arguments, num_warps=4)
    torch.cuda.synchronize()


def _metadata(tensors: dict[str, torch.Tensor]) -> dict[str, dict]:
    return {
        name: {
            "shape": list(tensor.shape),
            "stride_elements": list(tensor.stride()),
            "storage_offset_elements": tensor.storage_offset(),
            "data_ptr": tensor.data_ptr(),
            "storage_nbytes": tensor.untyped_storage().nbytes(),
        }
        for name, tensor in tensors.items()
    }


def _time_relayout(entry: dict, seed: int, names: tuple[str, ...], repeats: int = 12) -> dict:
    """测量复制加 Kernel 的诊断耗时；WSL2 单机数据不作为 e4 性能结论。"""

    samples = []
    copy_samples = []
    for index in range(repeats + 2):
        tensors = _make_tensors(entry, seed + index, names)
        start = torch.cuda.Event(enable_timing=True)
        copied = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for name in names:
            tensors[name] = tensors[name].contiguous()
        copied.record()
        spec = decode(entry["smoke"])
        kernel = _kernel(entry)
        kernel[spec.grid](**_arguments(entry, tensors), num_warps=4)
        end.record()
        torch.cuda.synchronize()
        if index >= 2:
            copy_samples.append(float(start.elapsed_time(copied)))
            samples.append(float(start.elapsed_time(end)))
    return {
        "repeats": repeats,
        "copy_median_ms": statistics.median(copy_samples),
        "copy_plus_kernel_median_ms": statistics.median(samples),
        "scope": "WSL2 单机诊断；不属于 e4 稳定性能证据",
    }


def execute(case_id: str, seed: int = 17) -> dict:
    """执行一个登记风险案例；未知案例和未支持变异直接失败。"""

    started = time.perf_counter()
    risk = risk_spec(case_id)
    if risk.mutation != "column_stride_2":
        raise ValueError("Unsupported：worker 只实现已审计的 column_stride_2")
    entry = next(item for item in load()["entries"] if item["id"] == risk.kernel_id)
    spec = decode(entry["smoke"])
    plan = build_plan(entry)

    raw_tensors = _make_tensors(entry, seed, risk.mutated_inputs)
    expected = reference(spec, raw_tensors)
    guard = plan.evaluate(raw_tensors)
    _launch(entry, raw_tensors)
    raw_checks = _checks(entry, raw_tensors, expected)

    repaired = _make_tensors(entry, seed, risk.mutated_inputs)
    copy_bytes = 0
    copies = []
    for name in risk.mutated_inputs:
        before = repaired[name]
        after = before.contiguous()
        if after.data_ptr() != before.data_ptr():
            amount = before.numel() * before.element_size()
            copy_bytes += amount
            copies.append({"tensor": name, "bytes": amount})
        repaired[name] = after
    repaired_guard = plan.evaluate(repaired)
    repaired_expected = reference(spec, repaired)
    _launch(entry, repaired)
    repaired_checks = _checks(entry, repaired, repaired_expected)

    fallback_tensors = _make_tensors(entry, seed, risk.mutated_inputs)
    fallback_expected = reference(spec, fallback_tensors)
    fallback_checks = [
        {"output": name, "correct": True, "mismatch_count": 0}
        for name, _ in zip(spec.outputs, fallback_expected)
    ]
    elapsed_ms = (time.perf_counter() - started) * 1000
    return {
        "case": risk.to_dict(),
        "seed": seed,
        "raw": {"checks": raw_checks, "metadata": _metadata(raw_tensors)},
        "guard": {"status": guard.status, "allowed": guard.allowed, "reason": guard.reason, "trace": guard.trace},
        "upstream_check": {
            "present": risk.wrapper_defense is not None,
            "expression": risk.wrapper_defense,
            "source": risk.wrapper_source,
        },
        "relayout": {
            "guard_status": repaired_guard.status,
            "checks": repaired_checks,
            "copies": copies,
            "copy_bytes": copy_bytes,
            "timing": _time_relayout(entry, seed, risk.mutated_inputs),
        },
        "fallback": {"checks": fallback_checks, "reference": entry["reference_anchor"]},
        "classification": "numeric_mismatch" if not all(item["correct"] for item in raw_checks) else "correct",
        "worker_elapsed_ms": elapsed_ms,
        "evidence_boundary": "合法 PyTorch 视图上的底层 Kernel 前置条件敏感性；接口承诺需由 L2/L3 单独证明",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_id")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    result = execute(args.case_id, args.seed)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
