"""运行并汇总 e3 的 L1/L2/L3 真实风险证据。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

from bench.e3.specs import RISK_SPECS


PROJECT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT / "results"


def _run(case_id: str, seed: int, timeout: int) -> dict:
    command = [sys.executable, "-m", "bench.e3.worker", case_id, "--seed", str(seed)]
    try:
        done = subprocess.run(command, cwd=PROJECT, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return {"case_id": case_id, "status": "Unknown", "reason": "worker_timeout", "exit_code": None}
    lines = [line for line in done.stdout.splitlines() if line.strip()]
    if done.returncode != 0 or len(lines) != 1:
        return {
            "case_id": case_id,
            "status": "Unknown",
            "reason": "worker_failure",
            "exit_code": done.returncode,
            "stdout": done.stdout[-2000:],
            "stderr": done.stderr[-4000:],
        }
    try:
        detail = json.loads(lines[0])
    except json.JSONDecodeError:
        return {"case_id": case_id, "status": "Unknown", "reason": "invalid_worker_json", "exit_code": done.returncode}
    if detail.get("case", {}).get("id") != case_id:
        return {"case_id": case_id, "status": "Unknown", "reason": "worker_identity_mismatch", "exit_code": done.returncode}
    return {"case_id": case_id, "status": "complete", "exit_code": done.returncode, "detail": detail}


def build(*, seed: int = 17, timeout: int = 180) -> dict:
    """每个案例使用独立进程；故障保留为 Unknown，不静默转成风险见证。"""

    rows = [_run(item.id, seed, timeout) for item in RISK_SPECS]
    complete = [row for row in rows if row["status"] == "complete"]
    witnesses = [row for row in complete if row["detail"]["classification"] == "numeric_mismatch"]
    l2 = [row for row in witnesses if row["detail"]["upstream_check"]["present"]]
    l3 = [row for row in witnesses if row["detail"]["case"]["l3_url"]]
    checks = {
        "all_workers_complete": len(complete) == len(RISK_SPECS),
        "independent_reference_for_every_case": all(
            row["detail"]["fallback"]["reference"] and all(item["correct"] for item in row["detail"]["fallback"]["checks"])
            for row in complete
        ),
        "raw_risk_reproduced": len(witnesses) >= 3,
        "two_sources_three_patterns": len({row["detail"]["case"]["source"] for row in witnesses}) >= 2
        and len({row["detail"]["case"]["problem"] for row in witnesses}) >= 3,
        "guard_blocks_every_witness": all(not row["detail"]["guard"]["allowed"] for row in witnesses),
        "relayout_and_fallback_correct": all(
            row["detail"]["relayout"]["guard_status"] == "True"
            and all(item["correct"] for item in row["detail"]["relayout"]["checks"])
            and all(item["correct"] for item in row["detail"]["fallback"]["checks"])
            for row in witnesses
        ),
        "l2_requires_fixed_wrapper_source": all(
            row["detail"]["upstream_check"]["source"] and row["detail"]["relayout"]["copy_bytes"] > 0 for row in l2
        ),
        "l3_requires_public_url_and_status": all(
            row["detail"]["case"]["l3_url"].startswith("https://github.com/")
            and row["detail"]["case"]["l3_status"]
            and row["detail"]["case"]["l3_commit"]
            for row in l3
        ),
    }
    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "seed": seed,
        "worker_timeout_seconds": timeout,
        "rows": rows,
        "counts": {
            "registered": len(RISK_SPECS),
            "complete": len(complete),
            "l1_witnesses": len(witnesses),
            "l2_wrapper_defenses": len(l2),
            "l3_public_items": len({row["detail"]["case"]["l3_url"] for row in l3}),
            "by_source": dict(Counter(row["detail"]["case"]["source"] for row in witnesses)),
        },
        "checks": checks,
        "go_risk": all(checks.values()),
        "scope": {
            "l1": "底层 Kernel 对合法 PyTorch 物理布局的前置条件敏感性；没有接口承诺时不称为上游漏洞",
            "l2": "固定源码中真实存在的 contiguous 防御、复制字节与 WSL2 诊断耗时",
            "l3": "截至 2026-10-02 可公开核对的 Issue/PR；只登记与本地问题模式语义对应的项目",
        },
    }


def write(report: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    base = RESULTS / "e3_real_risk_cases"
    base.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# e3 真实风险案例",
        "",
        f"- 生成时间：{report['generated_at']}。",
        f"- 状态：`{'go' if report['go_risk'] else 'no_go'}`；计数：`{report['counts']}`。",
        "- L1、L2、L3 分层互不替代；WSL2 复制耗时仅作诊断，不属于 e4 性能结论。",
        "",
        "| 案例 | 来源 | 原始调用 | Guard | Relayout | 复制字节 | L2 | L3 |",
        "| --- | --- | --- | --- | --- | ---: | --- | --- |",
    ]
    for row in report["rows"]:
        if row["status"] != "complete":
            lines.append(f"| {row['case_id']} | Unknown | {row['reason']} | Unknown | Unknown | 0 | 否 | 否 |")
            continue
        detail = row["detail"]
        case = detail["case"]
        lines.append(
            f"| `{case['id']}` | {case['source']} | {detail['classification']} | {detail['guard']['status']} | "
            f"{'correct' if all(item['correct'] for item in detail['relayout']['checks']) else 'failed'} | "
            f"{detail['relayout']['copy_bytes']} | {'是' if detail['upstream_check']['present'] else '否'} | "
            f"{case['l3_url'] or '0'} |"
        )
    lines.extend(["", "## 边界", ""])
    lines.extend(f"- **{level.upper()}**：{text}" for level, text in report["scope"].items())
    lines.append("")
    base.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    report = build(seed=args.seed, timeout=args.timeout)
    write(report)
    print(json.dumps({"go_risk": report["go_risk"], "counts": report["counts"]}, ensure_ascii=False))
    return 0 if report["go_risk"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

