"""第五阶段离线成本标定；在线分派不调用本模块。"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch
import triton

from pact.cost_model import COST_VERSION, CostEntry, layout_family, repair_fingerprint, shape_bucket
from pact.guard_plan import RULE_VERSION, compile_guard_plan
from pact.relayout import try_relayout
from scenarios.cases import reference
from scenarios.guard_worker import make_input


ROOT = Path(__file__).resolve().parent.parent
RECIPES = (
    {"name": "A", "layout": "contiguous"},
    {"name": "A", "layout": "transpose"},
    {"name": "A2", "layout": "padded"},
    {"name": "B", "layout": "contiguous", "n": 127},
    {"name": "B", "layout": "contiguous", "n": 128},
    {"name": "B", "layout": "offset", "n": 129},
    {"name": "D", "layout": "x_strided", "n": 129},
    {"name": "D", "layout": "y_strided", "n": 129},
    {"name": "holdout_vector", "layout": "strided", "n": 129},
    {"name": "holdout_matrix", "layout": "strided"},
)


def _stats(samples: list[float]) -> dict:
    ordered = sorted(samples)
    return {"samples_us": samples, "median_us": statistics.median(samples),
            "p95_us": ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))],
            "min_us": ordered[0], "max_us": ordered[-1]}


def _event(fn) -> float:
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000


def _sample_case(recipe: dict, warmup: int, repeats: int) -> tuple[dict, CostEntry]:
    value = make_input(recipe)
    x, y = value if isinstance(value, tuple) else (value, None)
    tensors = {"X": x, **({"Y": y} if y is not None else {})}
    plan = compile_guard_plan(recipe["name"])
    direct = plan.evaluate(tensors)
    repair = try_relayout(plan, tensors, direct) if direct.status == "False" and direct.failed else None
    feasible = ["PyTorch Fallback"]
    if direct.allowed:
        feasible.insert(0, "Fast")
    if repair and repair.status == "True":
        feasible.insert(0, "Relayout+Fast")

    def execute(path: str):
        guard = plan.evaluate(tensors)
        if path == "Fast":
            return plan.launch(tensors, guard)
        if path == "Relayout+Fast":
            rerun = try_relayout(plan, tensors, guard)
            if rerun.status != "True":
                raise RuntimeError("复制复验失败")
            return plan.launch(rerun.tensors, rerun.guard)
        return reference(x, y)

    timings = {}
    for path in feasible:
        for _ in range(warmup):
            execute(path)
        torch.cuda.synchronize()
        samples = []
        for _ in range(repeats):
            before = time.perf_counter_ns()
            execute(path)
            torch.cuda.synchronize()
            samples.append((time.perf_counter_ns() - before) / 1000)
        timings[path] = _stats(samples)

    # 组成时间与完整调用分别测量；各段之和不等于端到端时间。
    guard_samples = []
    allocation_samples = []
    copy_samples = []
    kernel_samples = []
    fallback_samples = []
    for _ in range(repeats):
        before = time.perf_counter_ns()
        plan.evaluate(tensors)
        guard_samples.append((time.perf_counter_ns() - before) / 1000)
        if repair and repair.status == "True":
            alloc_before = time.perf_counter_ns()
            targets = {name: torch.empty(tuple(tensors[name].shape), device=tensors[name].device, dtype=tensors[name].dtype) for name in repair.copied}
            allocation_samples.append((time.perf_counter_ns() - alloc_before) / 1000)
            copy_samples.append(_event(lambda: [targets[name].copy_(tensors[name]) for name in repair.copied]))
        chosen_tensors = repair.tensors if repair and repair.status == "True" else tensors
        chosen_guard = plan.evaluate(chosen_tensors)
        if chosen_guard.allowed:
            kernel_samples.append(_event(lambda: plan.launch(chosen_tensors, chosen_guard)))
        fallback_samples.append(_event(lambda: reference(x, y)))
    components = {"guard_host": _stats(guard_samples), "allocation_host": _stats(allocation_samples) if allocation_samples else None,
                  "copy_gpu_event": _stats(copy_samples) if copy_samples else None,
                  "kernel_gpu_event": _stats(kernel_samples) if kernel_samples else None,
                  "fallback_gpu_event": _stats(fallback_samples)}
    copied = repair.copied if repair and repair.status == "True" else ()
    ordered_paths = sorted(feasible, key=lambda path: timings[path]["median_us"])
    stable = len(ordered_paths) == 1 or timings[ordered_paths[0]]["p95_us"] < timings[ordered_paths[1]]["min_us"]
    key = (recipe["name"], shape_bucket(tuple(x.shape)), layout_family(tensors))
    entry = CostEntry(key, plan.fingerprint, str(x.dtype), str(x.device), copied,
                      tuple((name, tensor.data_ptr() % 16) for name, tensor in sorted(tensors.items())),
                      tuple(x.shape), tuple((path, timings[path]["median_us"]) for path in feasible), repeats, stable)
    return {"recipe": recipe, "key": key, "feasible": feasible, "direct_guard": direct.status,
            "copied": copied, "cost_order_stable": stable, "selected_if_loaded": ordered_paths[0] if stable else ("Fast" if direct.allowed else "PyTorch Fallback"),
            "end_to_end_host": timings, "components": components}, entry


def build(warmup: int, repeats: int) -> dict:
    cases, entries = [], []
    for recipe in RECIPES:
        case, entry = _sample_case(recipe, warmup, repeats)
        cases.append(case)
        entries.append(entry.to_dict())
    return {"generated_at": datetime.now().astimezone().isoformat(), "version": COST_VERSION,
            "guard_version": RULE_VERSION, "warmup": warmup, "repeats": repeats,
            "repair_fingerprint": repair_fingerprint(),
            "jit_compilation_excluded": True, "synchronization": "CUDA Event synchronize for GPU segments; torch.cuda.synchronize after each end-to-end call",
            "environment": {"python": platform.python_version(), "torch": torch.__version__, "triton": triton.__version__,
                            "gpu": torch.cuda.get_device_name(), "cuda": torch.version.cuda},
            "cases": cases, "table": entries,
            "limits": "同机同代码精确 shape/dtype/device/布局/地址余数/修复集合适用；跨环境、指纹或样例域变化失效"}


def render(data: dict) -> str:
    lines = ["# 第五阶段离线成本标定", "", f"生成时间：{data['generated_at']}",
             f"GPU：{data['environment']['gpu']}；PyTorch {data['environment']['torch']}；Triton {data['environment']['triton']}。",
             f"预热 {data['warmup']} 次、每路径 {data['repeats']} 次；JIT 编译排除。", "",
             "端到端采用同步后的主机时钟；组成段的 GPU 操作采用 CUDA Event，Guard 用主机时钟（直接 Fast 时含实际输出分配与 span 检查）。复制分配单列主机时间。各段之和不等于端到端时间。", "",
             "| 案例 | 布局 | 可行路径 | 成本选择 | 端到端 median / p95 (µs) | Guard median (µs) | Allocate host median (µs) | Copy Event median (µs) | Kernel Event median (µs) | Fallback Event median (µs) |",
             "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for case in data["cases"]:
        times = "; ".join(f"{path}: {value['median_us']:.1f}/{value['p95_us']:.1f}" for path, value in case["end_to_end_host"].items())
        comp = case["components"]
        fmt = lambda value: f"{value['median_us']:.1f}" if value else "—"
        lines.append(f"| {case['recipe']['name']} | {case['recipe']['layout']} | {', '.join(case['feasible'])} | {case['selected_if_loaded']}{'' if case['cost_order_stable'] else ' (波动重叠，默认)'} | {times} | {fmt(comp['guard_host'])} | {fmt(comp['allocation_host'])} | {fmt(comp['copy_gpu_event'])} | {fmt(comp['kernel_gpu_event'])} | {fmt(comp['fallback_gpu_event'])} |")
    lines += ["", "JSON 保留全部原始样本、分位数、指纹和完整表项。在线仅查表；缺失、过期或不适用时使用确定性默认路径。", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/cost_calibration.md")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=15)
    args = parser.parse_args()
    if not 1 <= args.warmup <= 20 or not 3 <= args.repeats <= 100:
        parser.error("warmup/repeats 范围无效")
    data = build(args.warmup, args.repeats)
    args.output.write_text(render(data), encoding="utf-8")
    args.output.with_suffix(".json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"cost cases: {len(data['cases'])}, repeats: {args.repeats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
