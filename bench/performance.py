"""第六阶段受限样例的离线 Guard 与同步端到端性能试运行。"""

from __future__ import annotations

import argparse
import gc
import json
import math
import platform
import statistics
import subprocess
import time
from datetime import datetime
from pathlib import Path

import torch
import triton

from bench.metrics import paired_round_interval, paired_speedups
from pact.cost_model import CostTable
from pact.dispatch import dispatch
from pact.guard_plan import compile_guard_plan
from pact.relayout import try_relayout
from scenarios.cases import reference
from scenarios.guard_worker import make_input


ROOT = Path(__file__).resolve().parent.parent
HOLDOUTS = (
    {"name": "A", "layout": "contiguous", "m": 9, "n": 15},
    {"name": "A", "layout": "transpose", "m": 9, "n": 15},
    {"name": "A2", "layout": "padded", "m": 5, "n": 13},
    {"name": "B", "layout": "contiguous", "n": 255},
    {"name": "B", "layout": "strided", "n": 257},
    {"name": "D", "layout": "x_strided", "n": 257},
    {"name": "holdout_vector", "layout": "strided", "n": 257},
    {"name": "pointer_hint", "layout": "offset", "n": 256},
    {"name": "A", "layout": "contiguous", "m": 512, "n": 512},
    {"name": "B", "layout": "contiguous", "n": 262143},
    {"name": "D", "layout": "contiguous", "n": 262143},
    {"name": "D", "layout": "x_strided", "n": 129, "value_shift": 0.25},
    {"name": "D", "layout": "y_strided", "n": 129, "value_shift": 0.25},
)


def stats(samples: list[float]) -> dict:
    if not samples or any(not math.isfinite(value) or value <= 0 for value in samples):
        raise ValueError("计时样本必须为有限正数")
    ordered = sorted(samples)
    return {"samples_us": samples, "median_us": statistics.median(samples),
            "p95_us": ordered[math.ceil(0.95 * len(ordered)) - 1],
            "min_us": ordered[0], "max_us": ordered[-1]}


def recipe_label(recipe: dict) -> str:
    size = f"{recipe.get('m', 7)}×{recipe['n']}" if recipe["name"] in {"A", "A2", "holdout_matrix"} else str(recipe["n"])
    shift = f" shift={recipe['value_shift']}" if "value_shift" in recipe else ""
    return f"{recipe['name']} {recipe['layout']} {size}{shift}"


def interleaved_order(names: tuple[str, ...], rounds: int) -> list[tuple[str, ...]]:
    if not names or rounds < 1:
        raise ValueError("路径或轮次为空")
    return [tuple(names[(j + i) % len(names)] for j in range(len(names))) if i % 2 == 0
            else tuple(names[(j + i) % len(names)] for j in reversed(range(len(names))))
            for i in range(rounds)]


def _inputs(recipe: dict):
    value = make_input(recipe)
    x, y = value if isinstance(value, tuple) else (value, None)
    if "value_shift" in recipe:
        x.add_(recipe["value_shift"])
        if y is not None:
            y.add_(2 * recipe["value_shift"])
    return x, y


def _raw(name: str, plan, x: torch.Tensor, y: torch.Tensor | None):
    if name in {"A", "A2", "B"}:
        return plan.spec.wrapper(name, x)
    return plan.spec.wrapper(x, y) if y is not None else plan.spec.wrapper(x)


def _same(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    torch.cuda.synchronize()
    return tuple(actual.shape) == tuple(expected.shape) and bool(torch.allclose(actual, expected, rtol=1e-5, atol=1e-5))


def device_state() -> dict:
    command = ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.gr,clocks.mem,power.draw,power.limit",
               "--format=csv,noheader,nounits"]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"available": False, "reason": type(error).__name__}
    return {"available": result.returncode == 0, "fields": ("temperature_c", "graphics_mhz", "memory_mhz", "power_w", "limit_w"),
            "values": result.stdout.strip().splitlines(), "reason": result.stderr.strip()[-300:] if result.returncode else ""}


def measure_case(recipe: dict, *, warmup: int, repeats: int, batch_calls: int, cost_table: CostTable) -> dict:
    x, y = _inputs(recipe)
    name = recipe["name"]
    tensors = {"X": x, **({"Y": y} if y is not None else {})}
    plan = compile_guard_plan(name)
    if plan.status != "Supported":
        return {"recipe": recipe, "status": "Unsupported", "reason": plan.reason}
    expected = reference(x, y)
    direct = plan.evaluate(tensors)
    repair = try_relayout(plan, tensors, direct) if direct.status == "False" and direct.failed else None
    def fallback():
        return reference(x, y)
    def dispatch_default():
        return dispatch(name, x, y).output
    def dispatch_table():
        return dispatch(name, x, y, cost_table=cost_table).output
    def raw_fast():
        return _raw(name, plan, x, y)
    def repaired_fast():
        current = plan.evaluate(tensors)
        fixed = try_relayout(plan, tensors, current)
        if fixed.status != "True":
            raise RuntimeError("修复二次复验未通过")
        return plan.launch(fixed.tensors, fixed.guard)
    paths = {"Fallback": fallback, "DispatchDefault": dispatch_default, "DispatchTable": dispatch_table}
    if direct.allowed:
        paths["DirectFast"] = raw_fast
    if repair is not None and repair.status == "True":
        paths["RelayoutFast"] = repaired_fast
    for path, fn in paths.items():
        if not _same(fn(), expected):
            raise RuntimeError(f"{name}:{recipe['layout']} 的 {path} 与参考数值不一致")
    for _ in range(warmup):
        for fn in paths.values():
            fn()
    torch.cuda.synchronize()
    samples = {name: [] for name in paths}
    orders = interleaved_order(tuple(paths), repeats)
    batched = {name: [] for name in paths}
    enabled = gc.isenabled()
    if enabled:
        gc.disable()
    try:
        for order in orders:
            for path in order:
                before = time.perf_counter_ns()
                paths[path]()
                torch.cuda.synchronize()
                samples[path].append((time.perf_counter_ns() - before) / 1000)
        # 另测多次调用后一次同步的摊销吞吐成本；不把它叫作单次完成延迟。
        for order in orders:
            for path in order:
                outputs = []
                before = time.perf_counter_ns()
                for _ in range(batch_calls):
                    outputs.append(paths[path]())
                torch.cuda.synchronize()
                batched[path].append((time.perf_counter_ns() - before) / (1000 * batch_calls))
                outputs.clear()
    finally:
        if enabled:
            gc.enable()
    table_result = dispatch(name, x, y, cost_table=cost_table)
    return {"recipe": recipe, "status": "measured", "direct_guard": direct.status,
            "table_path": table_result.path, "table_reason": table_result.selection_reason,
            "repair_feasible": bool(repair and repair.status == "True"),
            "gc_disabled_during_measurement": True,
            "paths": {name: {"safe": True, **stats(values)} for name, values in samples.items()},
            "amortized_batch_host": {name: stats(values) for name, values in batched.items()},
            "order": orders, "same_semantics": True,
            "input_metadata": {key: {"shape": tuple(t.shape), "stride": tuple(t.stride()),
                                      "offset": t.storage_offset(), "ptr_mod16": t.data_ptr() % 16} for key, t in tensors.items()},
            "fingerprint": plan.fingerprint}


def measure_guard(*, warmup: int, repeats: int) -> dict:
    cases = ({"name": "B", "layout": "contiguous", "n": 255},
             {"name": "B", "layout": "strided", "n": 257},
             {"name": "pointer_hint", "layout": "offset", "n": 256})
    records = []
    for recipe in cases:
        x, _ = _inputs(recipe)
        plan = compile_guard_plan(recipe["name"])
        for _ in range(warmup):
            plan.evaluate({"X": x})
        enabled = gc.isenabled()
        if enabled:
            gc.disable()
        try:
            samples = []
            for _ in range(repeats):
                before = time.perf_counter_ns()
                result = plan.evaluate({"X": x})
                samples.append((time.perf_counter_ns() - before) / 1000)
        finally:
            if enabled:
                gc.enable()
        records.append({"recipe": recipe, "status": result.status,
                        "stages": [item["stage"] for item in result.checks], "host": stats(samples)})
    return {"cases": records, "gc_disabled_during_measurement": True,
            "note": "完整 GuardPlan.evaluate 含实际输出分配；不是纯谓词 CPU 纳秒级测量。"}


def build(*, warmup: int = 3, repeats: int = 15, batch_calls: int = 16) -> dict:
    if not 1 <= warmup <= 100 or not 3 <= repeats <= 200 or not 2 <= batch_calls <= 128:
        raise ValueError("预热或重复轮数无效")
    cost_table = CostTable.from_report(ROOT / "results/cost_calibration.json")
    state_before = device_state()
    guard = measure_guard(warmup=warmup, repeats=repeats)
    rows = [measure_case(recipe, warmup=warmup, repeats=repeats, batch_calls=batch_calls, cost_table=cost_table) for recipe in HOLDOUTS]
    state_after = device_state()
    speedup = paired_speedups([{"id": str(row["recipe"]), **row} for row in rows], "Fallback", "DispatchTable")
    interval = paired_round_interval(rows, "Fallback", "DispatchTable")
    batch_rows = [{"id": str(row["recipe"]), "same_semantics": row.get("same_semantics", False),
                   "paths": {name: {"safe": row["paths"][name]["safe"], "median_us": value["median_us"]}
                             for name, value in row.get("amortized_batch_host", {}).items()}}
                  for row in rows]
    batch_speedup = paired_speedups(batch_rows, "Fallback", "DispatchTable")
    batch_interval = paired_round_interval([{"same_semantics": row.get("same_semantics", False),
                                             "paths": {name: {"safe": True, "samples_us": value["samples_us"]}
                                                       for name, value in row.get("amortized_batch_host", {}).items()}}
                                            for row in rows], "Fallback", "DispatchTable")
    return {"generated_at": datetime.now().astimezone().isoformat(), "phase": "six_limited_holdout",
            "environment": {"python": platform.python_version(), "torch": torch.__version__,
                            "triton": triton.__version__, "cuda": torch.version.cuda,
                            "gpu": torch.cuda.get_device_name(), "cost_version": cost_table.version,
                            "platform": platform.platform(), "device_state_before": state_before, "device_state_after": state_after},
            "warmup": warmup, "repeats": repeats, "batch_calls": batch_calls, "jit_excluded": True,
            "protocol": "各安全路径逐轮交错且轮换顺序；每次调用后 torch.cuda.synchronize；主机 perf_counter_ns。输入已构造，输出分配留在调用内。",
            "guard": guard, "rows": rows, "speedup_vs_fallback": speedup,
            "paired_round_interval": interval,
            "amortized_batch_speedup_vs_fallback": batch_speedup, "amortized_batch_interval": batch_interval,
            "limits": "仅历史受限入口的 11 组留出尺寸与 2 组同元数据留出数值；缺失新算子、新来源、真实 SMT 删除消融与独立重复硬件，不代表第六阶段完整基准。"}


def render(data: dict) -> str:
    lines = ["# 第六阶段受限 Guard 与分派性能试运行", "", f"生成时间：{data['generated_at']}",
             f"环境：{data['environment']['gpu']}；Torch {data['environment']['torch']}；Triton {data['environment']['triton']}。",
             f"测量前设备状态：{data['environment']['device_state_before']}；测量后：{data['environment']['device_state_after']}。",
             f"预热 {data['warmup']}，交错测量 {data['repeats']} 轮；JIT 编译排除。", "",
             "| 输入 | 直接 Guard | 成本表实际路径 | 各安全路径 p50/p95 (µs) |", "| --- | --- | --- | --- |"]
    for row in data["rows"]:
        if row["status"] != "measured":
            lines.append(f"| {row['recipe']} | {row['status']} | — | {row['reason']} |")
            continue
        timings = "; ".join(f"{name}: {value['median_us']:.1f}/{value['p95_us']:.1f}" for name, value in row["paths"].items())
        lines.append(f"| {recipe_label(row['recipe'])} | {row['direct_guard']} | {row['table_path']} | {timings} |")
    result = data["speedup_vs_fallback"]
    batched = data["amortized_batch_speedup_vs_fallback"]
    ci = data["paired_round_interval"]
    batch_ci = data["amortized_batch_interval"]
    table_hits = sum(row.get("table_reason") == "匹配离线成本条目" for row in data["rows"])
    lines += ["", f"成对安全输入 {result['paired_count']} 个；单次同步 Fallback/DispatchTable 几何平均比 {result['geomean']}；不确定性需结合原始样本判断。",
              f"按轮次配对 bootstrap 比值 {ci['point_geomean']}，条件 95% 区间 {ci['ci95']}。",
              f"每批 {data['batch_calls']} 次调用、一次同步的摊销吞吐比：{batched['geomean']}（非单次调用延迟）。",
              f"批量摊销配对 bootstrap 比值 {batch_ci['point_geomean']}，条件 95% 区间 {batch_ci['ci95']}。",
              f"当前成本表精确匹配：{table_hits}/{len(data['rows'])}；未命中时执行第五阶段的确定性默认路径。",
              "设备状态只在测量前后读取；图形频率或温度变化可能使稳态假设失效。条件区间不覆盖跨运行和跨硬件波动。",
              "", "## 摊销批量主机成本 p50/p95 (µs/调用)", "", "| 输入 | 各安全路径 |", "| --- | --- |"]
    for row in data["rows"]:
        if row["status"] == "measured":
            timings = "; ".join(f"{name}: {value['median_us']:.1f}/{value['p95_us']:.1f}" for name, value in row["amortized_batch_host"].items())
            lines.append(f"| {recipe_label(row['recipe'])} | {timings} |")
    lines += [
              "", "## GuardPlan.evaluate 主机时间", "", "| 输入 | 判定 | p50/p95 (µs) | 已运行阶段 |", "| --- | --- | --- | --- |"]
    for row in data["guard"]["cases"]:
        lines.append(f"| {recipe_label(row['recipe'])} | {row['status']} | {row['host']['median_us']:.1f}/{row['host']['p95_us']:.1f} | {', '.join(row['stages'])} |")
    lines += ["", data["protocol"], data["guard"]["note"], data["limits"], ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/phase6_dispatch_performance.md")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--batch-calls", type=int, default=16)
    parser.add_argument("--rerender-existing", action="store_true", help="只从同名 JSON 更新 Markdown，不再次运行 GPU")
    args = parser.parse_args()
    data = (json.loads(args.output.with_suffix(".json").read_text(encoding="utf-8")) if args.rerender_existing
            else build(warmup=args.warmup, repeats=args.repeats, batch_calls=args.batch_calls))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(data), encoding="utf-8")
    args.output.with_suffix(".json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"measured={sum(row['status'] == 'measured' for row in data['rows'])} paired={data['speedup_vs_fallback']['paired_count']}")
    return 0 if all(row["status"] == "measured" for row in data["rows"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
