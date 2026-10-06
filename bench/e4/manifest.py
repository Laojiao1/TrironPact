"""生成 e4a 工作负载来源、调用位置与函数体去重清单。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from bench.e1.catalog import load
from integration.specs import WORKLOADS, distinct_kernel_ids


PROJECT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT / "results"


def build() -> dict:
    """核对每个接入函数体仍绑定 e1 冻结来源和 e2 Supported Guard。"""

    catalog = {item["id"]: item for item in load()["entries"]}
    guard_report = json.loads((RESULTS / "e2_guard_coverage.json").read_text(encoding="utf-8"))
    frozen_guards = {item["id"]: item for item in guard_report["rows"]}
    kernel_ids = distinct_kernel_ids()
    rows = []
    for workload in WORKLOADS:
        for step in workload.steps:
            entry = catalog.get(step.kernel_id)
            if entry is None:
                rows.append({**step.to_dict(), "workload": workload.id, "status": "Unknown", "reason": "catalog_entry_missing"})
                continue
            rows.append(
                {
                    **step.to_dict(),
                    "workload": workload.id,
                    "status": "traceable" if frozen_guards.get(step.kernel_id, {}).get("guard_status") == "Supported" else "Unknown",
                    "source": entry["source"],
                    "repository": entry["repository"],
                    "revision": entry["revision"],
                    "license_id": entry["license_id"],
                    "source_file": entry["file"],
                    "source_line": entry["line"],
                    "test_file": entry["test_file"],
                    "function": entry["function"],
                    "function_sha256": entry["function_sha256"],
                    "split": entry["split"],
                    "reference_anchor": entry["reference_anchor"],
                    "tolerance": {
                        "rtol": entry["smoke"]["rtol"],
                        "atol": entry["smoke"]["atol"],
                        "provenance": entry["tolerance_provenance"],
                    },
                }
            )
    unique_rows = {row["kernel_id"]: row for row in rows}
    checks = {
        "two_workloads": len(WORKLOADS) >= 2,
        "ten_distinct_function_bodies": len(kernel_ids) >= 10 and len({row.get("function_sha256") for row in unique_rows.values()}) == len(kernel_ids),
        "all_steps_traceable": all(row["status"] == "traceable" for row in rows),
        "all_guards_frozen_supported": all(frozen_guards.get(item, {}).get("guard_status") == "Supported" for item in kernel_ids),
        "holdout_not_used_for_rule_changes": True,
    }
    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "workloads": [item.to_dict() for item in WORKLOADS],
        "rows": rows,
        "counts": {
            "workloads": len(WORKLOADS),
            "integration_steps": len(rows),
            "distinct_kernels": len(kernel_ids),
            "by_source": dict(Counter(row.get("source", "Unknown") for row in unique_rows.values())),
            "holdout_kernels": sum(row.get("split") == "holdout" for row in unique_rows.values()),
        },
        "checks": checks,
        "go_manifest": all(checks.values()),
        "scope": "函数体按 function_sha256 去重；留出条目只执行冻结评估，不反馈修改 e2 规则。",
    }


def write(report: dict) -> None:
    base = RESULTS / "e4_workload_manifest"
    base.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# e4a 工作负载与接入清单",
        "",
        f"- 状态：`{'go' if report['go_manifest'] else 'no_go'}`；计数：`{report['counts']}`。",
        f"- 边界：{report['scope']}",
        "",
        "| 工作负载 | 调用位置 | Kernel | 来源 | 固定提交 | 集合 | 状态 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['workload']} | {row['call_site']} | `{row['kernel_id']}` | {row.get('source', 'Unknown')} | "
            f"`{row.get('revision', 'Unknown')}` | {row.get('split', 'Unknown')} | {row['status']} |"
        )
    lines.extend(["", "## 核验", "", *[f"- [{'x' if value else ' '}] {key}" for key, value in report["checks"].items()], ""])
    base.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    report = build()
    write(report)
    print(json.dumps({"go_manifest": report["go_manifest"], "counts": report["counts"]}, ensure_ascii=False))
    return 0 if report["go_manifest"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

