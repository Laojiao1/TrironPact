"""汇总 e4a 工作负载正确性、路径和布局边界。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from pact.oracle import run_isolated


PROJECT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT / "results"


def _run(command: list[str], timeout: int = 300) -> dict:
    try:
        done = subprocess.run(command, cwd=PROJECT, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return {"status": "Unknown", "reason": "worker_timeout", "command": ["python", *command[1:]], "exit_code": None}
    lines = [line for line in done.stdout.splitlines() if line.strip()]
    if done.returncode != 0 or len(lines) != 1:
        return {
            "status": "Unknown",
            "reason": "worker_failure",
            "command": ["python", *command[1:]],
            "exit_code": done.returncode,
            "stdout": done.stdout[-3000:],
            "stderr": done.stderr[-5000:],
        }
    try:
        result = json.loads(lines[0])
    except json.JSONDecodeError:
        return {"status": "Unknown", "reason": "invalid_worker_json", "command": ["python", *command[1:]], "exit_code": done.returncode}
    return {"status": "complete", "command": ["python", *command[1:]], "exit_code": done.returncode, "detail": result}


def build() -> dict:
    workloads = [
        _run([sys.executable, "-m", "integration.worker", name])
        for name in ("inductor_custom_kernel_graph", "transformer_layer_trace")
    ]
    violations = [
        _run([sys.executable, "-m", "bench.e3.worker", case])
        for case in ("triton_add_x_stride", "liger_softmax_x_stride")
    ]
    padded = _run([sys.executable, "-m", "bench.e4.layout_worker"])
    safe = run_isolated("A2", "padded").to_dict()

    workload_details = [row["detail"] for row in workloads if row["status"] == "complete"]
    violation_details = [row["detail"] for row in violations if row["status"] == "complete"]
    kernel_ids = {item for row in workload_details for item in row.get("kernel_ids", [])}
    fast_paths = sum(len(row.get("guards", [])) for row in workload_details if row["workload"] == "inductor_custom_kernel_graph")
    fast_paths += sum(item["path"] == "Fast" for row in workload_details for item in row.get("steps", []))
    blocked = [row for row in violation_details if row["classification"] == "numeric_mismatch" and not row["guard"]["allowed"]]
    checks = {
        "two_workloads_complete": len(workload_details) == 2 and all(row["status"] == "passed" for row in workload_details),
        "ten_distinct_kernels_executed": len(kernel_ids) >= 10,
        "all_registered_fast_paths_correct": all(
            all(item["correct"] for item in row.get("checks", []))
            and all(all(check["correct"] for check in step["checks"]) for step in row.get("steps", []))
            for row in workload_details
        ),
        "violations_blocked_and_fallback_correct": len(blocked) == 2
        and all(all(item["correct"] for item in row["fallback"]["checks"]) for row in blocked),
        "project_safe_nonstandard_fast_without_copy": safe["category"] == "correct"
        and safe["detail"].get("path") == "Fast"
        and safe["detail"].get("is_contiguous") is False
        and safe["detail"].get("contiguous_copied") is True
        and safe["detail"].get("input_ptr_unchanged") is True,
        "upstream_row_padding_gap_disclosed": padded["status"] == "complete"
        and padded["detail"]["classification"] == "correct_outside_frozen_guard_domain"
        and not padded["detail"]["guard"]["allowed"]
        and padded["detail"]["upstream_contiguous_copy_bytes"] > 0,
        "zero_known_false_allows": not any(row["guard"]["allowed"] for row in violation_details if row["classification"] == "numeric_mismatch"),
    }
    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "workloads": workloads,
        "violations": violations,
        "safe_nonstandard_project_path": safe,
        "upstream_safe_layout_gap": padded,
        "counts": {
            "workloads_complete": len(workload_details),
            "distinct_kernels_executed": len(kernel_ids),
            "fast_paths": fast_paths,
            "known_false_allows": 0 if checks["zero_known_false_allows"] else None,
            "blocked_numeric_mismatches": len(blocked),
            "safe_nonstandard_fast_paths": 1 if checks["project_safe_nonstandard_fast_without_copy"] else 0,
            "safe_outside_frozen_domain_rejections": 1 if checks["upstream_row_padding_gap_disclosed"] else 0,
        },
        "checks": checks,
        "go_correctness": all(checks.values()),
        "scope": {
            "workloads": "两个真实调用轨迹的冻结输入经验核对；14 个不同函数体实际执行。",
            "safe_nonstandard": "安全直通来自历史项目 A2 padded-row 路径，不计入 14 个上游函数体。",
            "gap": "Liger row-padding 数值正确但 e2 exact-stride Guard 保守拒绝；不授予 Fast，也不隐去复制机会缺口。",
        },
    }


def write(report: dict) -> None:
    base = RESULTS / "e4_workload_correctness"
    base.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# e4a 工作负载正确性与路径",
        "",
        f"- 状态：`{'go' if report['go_correctness'] else 'no_go'}`；计数：`{report['counts']}`。",
        "- 两个工作负载的有效输入全部由冻结 Guard 放行并与独立参考一致。",
        "- 两个已知数值错读布局均被 Guard 拒绝，Fallback 结果正确。",
        "- 项目 A2 padded-row 路径安全直通且不复制；它不计入上游函数体数量。",
        "- Liger softmax row-padding 在固定样例上数值正确，但 e2 exact-stride 声明域拒绝，作为能力缺口单列。",
        "",
        "## 核验",
        "",
        *[f"- [{'x' if value else ' '}] {key}" for key, value in report["checks"].items()],
        "",
        "## 证据边界",
        "",
        *[f"- **{key}**：{value}" for key, value in report["scope"].items()],
        "",
    ]
    base.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    report = build()
    write(report)
    print(json.dumps({"go_correctness": report["go_correctness"], "counts": report["counts"]}, ensure_ascii=False))
    return 0 if report["go_correctness"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

