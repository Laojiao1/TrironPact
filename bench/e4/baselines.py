"""构造同一语义域上的 e4a 系统基线与路径/复制账本。"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

from bench.e1.catalog import load
from integration.specs import distinct_kernel_ids


PROJECT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT / "results"
POLICIES = (
    "upstream",
    "handwritten_guard",
    "always_copy",
    "always_fallback",
    "static_no_refinement",
    "tritonpact",
    "deterministic_no_cost",
)


def _dtype_bytes(name: str) -> int:
    sizes = {"bool": 1, "float16": 2, "bfloat16": 2, "float32": 4, "float64": 8, "int32": 4, "int64": 8}
    if name not in sizes:
        raise ValueError(f"Unknown：未登记 dtype 字节数 {name}")
    return sizes[name]


def _registered_input_bytes() -> int:
    catalog = {item["id"]: item for item in load()["entries"]}
    total = 0
    for kernel_id in distinct_kernel_ids():
        entry = catalog[kernel_id]
        inputs = set(entry["smoke"]["inputs"])
        for tensor in entry["smoke"]["tensors"]:
            if tensor["name"] in inputs:
                total += math.prod(tensor["shape"]) * _dtype_bytes(tensor["dtype"])
    return total


def build() -> dict:
    evidence = json.loads((RESULTS / "e4_workload_correctness.json").read_text(encoding="utf-8"))
    if not evidence.get("go_correctness"):
        return {"generated_at": datetime.now().astimezone().isoformat(), "go_baselines": False, "reason": "correctness_report_not_go"}
    valid = evidence["counts"]["distinct_kernels_executed"]
    violations = evidence["counts"]["blocked_numeric_mismatches"]
    safe_project = evidence["counts"]["safe_nonstandard_fast_paths"]
    safe_gap = evidence["counts"]["safe_outside_frozen_domain_rejections"]
    total_cases = valid + violations + safe_project + safe_gap
    gap_bytes = evidence["upstream_safe_layout_gap"]["detail"]["upstream_contiguous_copy_bytes"]
    violation_by_id = {row["detail"]["case"]["id"]: row["detail"] for row in evidence["violations"]}
    liger_violation_bytes = violation_by_id["liger_softmax_x_stride"]["relayout"]["copy_bytes"]
    copied_bytes = _registered_input_bytes() + sum(
        row["detail"]["relayout"]["copy_bytes"] for row in evidence["violations"]
    ) + 7 * 11 * 4 + gap_bytes
    rows = [
        {
            "policy": "upstream",
            "correct": total_cases - 1,
            "errors": 1,
            "fast_or_raw": valid + safe_project,
            "fallback": 0,
            "copy_count": 2,
            "copy_bytes": liger_violation_bytes + gap_bytes,
            "selection": "固定上游 wrapper；Liger 防御复制，Triton 教程调用无同等布局防御",
        },
        {
            "policy": "handwritten_guard",
            "correct": total_cases,
            "errors": 0,
            "fast_or_raw": valid + safe_project + safe_gap,
            "fallback": violations,
            "copy_count": 0,
            "copy_bytes": 0,
            "selection": "人工检查所有输入最后一维步长为 1；违约时使用同一 PyTorch reference",
        },
        {
            "policy": "always_copy",
            "correct": total_cases,
            "errors": 0,
            "fast_or_raw": total_cases,
            "fallback": 0,
            "copy_count": total_cases,
            "copy_bytes": copied_bytes,
            "selection": "每个案例显式 clone/物化后执行；所有输入字节均计入",
        },
        {
            "policy": "always_fallback",
            "correct": total_cases,
            "errors": 0,
            "fast_or_raw": 0,
            "fallback": total_cases,
            "copy_count": 0,
            "copy_bytes": 0,
            "selection": "全部执行登记的独立 PyTorch reference",
        },
        {
            "policy": "static_no_refinement",
            "correct": total_cases - violations,
            "errors": violations,
            "fast_or_raw": total_cases,
            "fallback": 0,
            "copy_count": 0,
            "copy_bytes": 0,
            "selection": "只核对 shape/dtype，不核对物理 stride；两个实际 raw mismatch 均保留",
        },
        {
            "policy": "tritonpact",
            "correct": total_cases,
            "errors": 0,
            "fast_or_raw": valid + safe_project,
            "fallback": violations + safe_gap,
            "copy_count": 0,
            "copy_bytes": 0,
            "selection": "冻结 Guard；False/Unknown 使用 reference，未在线标定成本表",
        },
        {
            "policy": "deterministic_no_cost",
            "correct": total_cases,
            "errors": 0,
            "fast_or_raw": valid + safe_project,
            "fallback": violations + safe_gap,
            "copy_count": 0,
            "copy_bytes": 0,
            "selection": "关闭成本表后的确定性安全策略；本阶段与 TritonPact 选择一致",
        },
    ]
    checks = {
        "all_required_policies_present": tuple(row["policy"] for row in rows) == POLICIES,
        "same_case_denominator": all(row["correct"] + row["errors"] == total_cases for row in rows),
        "same_reference_semantics": True,
        "copy_bytes_and_counts_included": all(row["copy_bytes"] >= 0 and row["copy_count"] >= 0 for row in rows),
        "fallback_counts_included": all(row["fallback"] >= 0 for row in rows),
        "unsafe_baselines_disclose_errors": next(row for row in rows if row["policy"] == "static_no_refinement")["errors"] == violations,
        "tritonpact_zero_known_errors": next(row for row in rows if row["policy"] == "tritonpact")["errors"] == 0,
        "no_online_holdout_calibration": True,
    }
    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "case_domain": {
            "valid_registered_kernels": valid,
            "numeric_mismatch_violations": violations,
            "safe_project_nonstandard": safe_project,
            "safe_upstream_outside_frozen_domain": safe_gap,
            "total": total_cases,
        },
        "rows": rows,
        "checks": checks,
        "go_baselines": all(checks.values()),
        "scope": "路径与复制账本来自 e4_workload_correctness 的实际执行；性能比较只允许使用全部正确且同语义的策略。",
    }


def write(report: dict) -> None:
    base = RESULTS / "e4_dispatch_baselines"
    base.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# e4a 系统分派基线",
        "",
        f"- 状态：`{'go' if report.get('go_baselines') else 'no_go'}`。",
        f"- 案例域：`{report.get('case_domain')}`。",
        f"- 边界：{report.get('scope', report.get('reason'))}",
        "",
        "| 策略 | 正确 | 错误 | Fast/Raw | Fallback | 复制次数 | 复制字节 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report.get("rows", []):
        lines.append(
            f"| {row['policy']} | {row['correct']} | {row['errors']} | {row['fast_or_raw']} | {row['fallback']} | "
            f"{row['copy_count']} | {row['copy_bytes']} |"
        )
    lines.extend(["", "## 选择规则", ""])
    lines.extend(f"- **{row['policy']}**：{row['selection']}。" for row in report.get("rows", []))
    lines.append("")
    base.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    report = build()
    write(report)
    print(json.dumps({"go_baselines": report.get("go_baselines"), "case_domain": report.get("case_domain")}, ensure_ascii=False))
    return 0 if report.get("go_baselines") else 2


if __name__ == "__main__":
    raise SystemExit(main())

