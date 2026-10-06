"""e4a 单机 WSL2 诊断计时；不产生跨硬件性能结论。"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path

import torch
import triton

from bench.e1.catalog import load
from bench.e2.plans import build_plan
from integration.worker import _inductor_graph, _inductor_inputs


PROJECT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT / "results"
POLICIES = ("upstream", "handwritten_guard", "always_copy", "always_fallback", "static_no_refinement", "tritonpact", "deterministic_no_cost")
WARMUP = 3
ROUNDS = 20


def percentile(values: list[float], quantile: float) -> float:
    """对已排序相邻点线性插值；空样本返回 ValueError 而非伪造 0。"""

    if not values:
        raise ValueError("Unknown：性能样本为空")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def paired_ratio_interval(numerator: list[float], denominator: list[float], *, seed: int = 17, draws: int = 2000) -> dict:
    """返回配对中位比及 bootstrap 95% 区间；单位相同且越小越好。"""

    if len(numerator) != len(denominator) or not numerator or any(value <= 0 for value in denominator):
        raise ValueError("Unknown：配对样本长度或分母无效")
    ratios = [left / right for left, right in zip(numerator, denominator)]
    rng = random.Random(seed)
    medians = []
    for _ in range(draws):
        medians.append(statistics.median(ratios[rng.randrange(len(ratios))] for _ in ratios))
    return {
        "median_ratio": statistics.median(ratios),
        "ci95": [percentile(medians, 0.025), percentile(medians, 0.975)],
        "stable_direction": "lower" if percentile(medians, 0.975) < 1 else "higher" if percentile(medians, 0.025) > 1 else "uncertain",
    }


def _gpu_state() -> dict:
    command = ["nvidia-smi", "--query-gpu=timestamp,name,temperature.gpu,clocks.sm,power.draw", "--format=csv,noheader,nounits"]
    try:
        done = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    except (FileNotFoundError, PermissionError, subprocess.TimeoutExpired) as error:
        return {"status": "Unknown", "reason": type(error).__name__}
    return {"status": "available" if done.returncode == 0 else "Unknown", "command": command, "exit_code": done.returncode, "raw": done.stdout.strip() or done.stderr.strip()}


def _environment() -> dict:
    nvcc = shutil.which("nvcc")
    nvcc_version = "Unknown"
    if nvcc:
        done = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=10, check=False)
        nvcc_version = (done.stdout + done.stderr).strip().splitlines()[-1] if done.returncode == 0 else "Unknown"
    return {
        "system": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown",
        "wsl_interop": "microsoft" in platform.release().lower() or bool(os.environ.get("WSL_DISTRO_NAME")),
        "nvcc": nvcc or "Unknown",
        "nvcc_version": nvcc_version,
        "path_rule": "e4a 命令显式把 /usr/local/cuda/bin 放在 Windows PATH 之前",
    }


def _reference(inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    x, y, mask, matrix = inputs
    return torch.where(mask, (x - y) * 2 + y, (x - y) * 2) + y, matrix * 2


def _manual_guard(inputs: tuple[torch.Tensor, ...]) -> bool:
    return all(tensor.ndim == 0 or tensor.stride(-1) == 1 for tensor in inputs)


def _static_guard(inputs: tuple[torch.Tensor, ...]) -> bool:
    x, y, mask, matrix = inputs
    return x.shape == y.shape == mask.shape == (129,) and matrix.shape == (16, 16) and x.dtype == y.dtype == matrix.dtype == torch.float32 and mask.dtype == torch.bool


def _guard_context(inputs: tuple[torch.Tensor, ...]) -> tuple[list, list[dict[str, torch.Tensor]]]:
    catalog = {item["id"]: item for item in load()["entries"]}
    names = ("pytorch_sub", "pytorch_double", "pytorch_masked_add", "triton_add", "pytorch_double_strided")
    plans = [build_plan(catalog[name]) for name in names]
    x, y, mask, matrix = inputs
    a, b, c, d = (torch.empty_like(x) for _ in range(4))
    matrix_out = torch.empty_like(matrix)
    mappings = [
        {"x": x, "y": y, "out": a},
        {"x": a, "out": b},
        {"x": b, "y": y, "mask": mask, "out": c},
        {"x": c, "y": y, "out": d},
        {"x": matrix, "out": matrix_out},
    ]
    return plans, mappings


def _evaluate_plans(plans: list, mappings: list[dict[str, torch.Tensor]]) -> bool:
    return all(plan.evaluate(tensors).allowed for plan, tensors in zip(plans, mappings))


def _timed(callable_) -> float:
    torch.cuda.synchronize()
    started = time.perf_counter_ns()
    callable_()
    torch.cuda.synchronize()
    return (time.perf_counter_ns() - started) / 1_000_000


def build() -> dict:
    inputs = _inductor_inputs(17)
    plans, mappings = _guard_context(inputs)
    compiled = torch.compile(_inductor_graph, fullgraph=True, backend="inductor")
    expected = _reference(inputs)
    first_started = time.perf_counter()
    first = compiled(*inputs)
    torch.cuda.synchronize()
    first_ms = (time.perf_counter() - first_started) * 1000
    if not all(torch.equal(actual, wanted) for actual, wanted in zip(first, expected)):
        raise RuntimeError("e4a 诊断前正确性核对失败")

    def upstream():
        return compiled(*inputs)

    def handwritten_guard():
        return compiled(*inputs) if _manual_guard(inputs) else _reference(inputs)

    def always_copy():
        copied = tuple(tensor.clone() for tensor in inputs)
        return compiled(*copied)

    def always_fallback():
        return _reference(inputs)

    def static_no_refinement():
        return compiled(*inputs) if _static_guard(inputs) else _reference(inputs)

    def tritonpact():
        return compiled(*inputs) if _evaluate_plans(plans, mappings) else _reference(inputs)

    callables = {
        "upstream": upstream,
        "handwritten_guard": handwritten_guard,
        "always_copy": always_copy,
        "always_fallback": always_fallback,
        "static_no_refinement": static_no_refinement,
        "tritonpact": tritonpact,
        "deterministic_no_cost": tritonpact,
    }
    for _ in range(WARMUP):
        for name in POLICIES:
            callables[name]()
        torch.cuda.synchronize()

    state_before = _gpu_state()
    samples = {name: [] for name in POLICIES}
    orders = []
    rng = random.Random(17)
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(ROUNDS):
            order = list(POLICIES)
            rng.shuffle(order)
            orders.append(order)
            for name in order:
                samples[name].append(_timed(callables[name]))
    finally:
        if gc_was_enabled:
            gc.enable()
    state_after = _gpu_state()

    guard_samples = [_timed(lambda: _evaluate_plans(plans, mappings)) for _ in range(ROUNDS)]
    copy_samples = [_timed(lambda: tuple(tensor.clone() for tensor in inputs)) for _ in range(ROUNDS)]
    summaries = {
        name: {
            "p50_ms": percentile(values, 0.5),
            "p95_ms": percentile(values, 0.95),
            "samples_ms": values,
            "paired_to_upstream": paired_ratio_interval(values, samples["upstream"], seed=17 + index),
        }
        for index, (name, values) in enumerate(samples.items())
    }
    checks = {
        "native_wsl_cuda_toolchain_selected": str(_environment()["nvcc"]).startswith("/usr/local/cuda"),
        "warmup_and_interleaving_recorded": len(orders) == ROUNDS and all(set(order) == set(POLICIES) for order in orders),
        "raw_samples_complete": all(len(values) == ROUNDS and all(value > 0 for value in values) for values in samples.values()),
        "guard_copy_kernel_fallback_components_present": bool(guard_samples and copy_samples and summaries["upstream"] and summaries["always_fallback"]),
        "first_run_separated": first_ms > 0,
        "no_online_cost_calibration": True,
        "diagnostic_scope_only": True,
    }
    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "environment": _environment(),
        "protocol": {
            "workload": "inductor_custom_kernel_graph",
            "warmup": WARMUP,
            "rounds": ROUNDS,
            "order": "fixed-seed randomized interleaving",
            "synchronization": "torch.cuda.synchronize before and after every timed call",
            "gc": "disabled during measured rounds",
            "units": "milliseconds",
            "timing": "host wall clock includes Guard/copy/dispatch preparation and synchronized GPU completion",
        },
        "first_run_including_compile_ms": first_ms,
        "components": {
            "guard_only": {"p50_ms": percentile(guard_samples, 0.5), "p95_ms": percentile(guard_samples, 0.95), "samples_ms": guard_samples},
            "copy_only": {"p50_ms": percentile(copy_samples, 0.5), "p95_ms": percentile(copy_samples, 0.95), "samples_ms": copy_samples},
            "kernel_graph": summaries["upstream"],
            "fallback": summaries["always_fallback"],
        },
        "policies": summaries,
        "interleaved_orders": orders,
        "gpu_state_before": state_before,
        "gpu_state_after": state_after,
        "checks": checks,
        "go_diagnostics": all(checks.values()),
        "conclusion": "WSL2 单机诊断；无论区间方向如何，均不形成稳定加速、原生 Linux 或跨硬件结论。",
    }


def write(report: dict) -> None:
    base = RESULTS / "e4_wsl_diagnostics"
    base.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# e4a WSL2 单机诊断",
        "",
        f"- 状态：`{'go' if report['go_diagnostics'] else 'no_go'}`。",
        f"- 首次运行（含编译）：{report['first_run_including_compile_ms']:.3f} ms。",
        f"- 结论：{report['conclusion']}",
        "",
        "| 策略 | p50 ms | p95 ms | 配对中位比 | 95% CI | 方向 |",
        "| --- | ---: | ---: | ---: | --- | --- |",
    ]
    for name, row in report["policies"].items():
        paired = row["paired_to_upstream"]
        lines.append(
            f"| {name} | {row['p50_ms']:.6f} | {row['p95_ms']:.6f} | {paired['median_ratio']:.4f} | "
            f"[{paired['ci95'][0]:.4f}, {paired['ci95'][1]:.4f}] | {paired['stable_direction']} |"
        )
    lines.extend(
        [
            "",
            "## 分项",
            "",
            f"- Guard：p50 {report['components']['guard_only']['p50_ms']:.6f} ms，p95 {report['components']['guard_only']['p95_ms']:.6f} ms。",
            f"- 显式复制：p50 {report['components']['copy_only']['p50_ms']:.6f} ms，p95 {report['components']['copy_only']['p95_ms']:.6f} ms。",
            f"- Kernel 图：p50 {report['components']['kernel_graph']['p50_ms']:.6f} ms，p95 {report['components']['kernel_graph']['p95_ms']:.6f} ms。",
            f"- Fallback：p50 {report['components']['fallback']['p50_ms']:.6f} ms，p95 {report['components']['fallback']['p95_ms']:.6f} ms。",
            "",
            "原始样本、交错顺序、GPU 前后状态和完整环境指纹保存在同名 JSON。",
            "",
        ]
    )
    base.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    report = build()
    write(report)
    print(json.dumps({"go_diagnostics": report["go_diagnostics"], "first_run_ms": report["first_run_including_compile_ms"], "conclusion": report["conclusion"]}, ensure_ascii=False))
    return 0 if report["go_diagnostics"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

