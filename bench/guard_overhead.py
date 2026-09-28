"""从同次第六阶段同步性能原始样本派生 Guard 绝对开销报告。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bench.performance import ROOT, recipe_label


def build(performance: dict) -> dict:
    rows = []
    for item in performance["rows"]:
        paths = item.get("paths", {})
        if "DirectFast" not in paths or "DispatchDefault" not in paths:
            continue
        raw = paths["DirectFast"]["samples_us"]
        dispatched = paths["DispatchDefault"]["samples_us"]
        if len(raw) != len(dispatched) or not raw:
            raise ValueError("成对样本长度不匹配")
        rows.append({"recipe": item["recipe"], "direct_fast": paths["DirectFast"],
                     "guarded_dispatch": paths["DispatchDefault"],
                     "paired_round_differences_us": [b - a for a, b in zip(raw, dispatched)],
                     "median_ratio": paths["DispatchDefault"]["median_us"] / paths["DirectFast"]["median_us"]})
    return {"source_generated_at": performance["generated_at"], "environment": performance["environment"],
            "guard_evaluate": performance["guard"], "direct_fast_pairs": rows,
            "limits": "GuardPlan.evaluate 含输出分配和 span；DirectFast 为仅在静态安全输入运行的原始 wrapper。同步与主机噪声使差值不等于独立谓词成本，不跨输入推断。"}


def render(data: dict) -> str:
    lines = ["# 第六阶段 Guard 开销受限试运行", "", f"原始测量时间：{data['source_generated_at']}",
             "", "| 输入 | GuardPlan.evaluate p50/p95 (µs) | 判定 |", "| --- | ---: | --- |"]
    for item in data["guard_evaluate"]["cases"]:
        lines.append(f"| {recipe_label(item['recipe'])} | {item['host']['median_us']:.1f}/{item['host']['p95_us']:.1f} | {item['status']} |")
    lines += ["", "| 输入 | 原始安全 Fast p50 (µs) | 完整分派 p50 (µs) | p50 比 |", "| --- | ---: | ---: | ---: |"]
    for item in data["direct_fast_pairs"]:
        lines.append(f"| {recipe_label(item['recipe'])} | {item['direct_fast']['median_us']:.1f} | {item['guarded_dispatch']['median_us']:.1f} | {item['median_ratio']:.3f} |")
    lines += ["", data["limits"], ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "results/phase6_dispatch_performance.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/phase6_guard_overhead.md")
    args = parser.parse_args()
    data = build(json.loads(args.source.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(data), encoding="utf-8")
    args.output.with_suffix(".json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"guard_cases={len(data['guard_evaluate']['cases'])} direct_pairs={len(data['direct_fast_pairs'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
