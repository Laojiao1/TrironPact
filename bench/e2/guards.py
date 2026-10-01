"""生成 e2 Guard 可用性与 Fast 可行输入报告。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime

from bench.e1.catalog import load
from bench.e2.coverage import PROJECT, RESULTS
from bench.e2.plans import build_plan


def _entries(evaluate_holdout: bool) -> list[dict]:
    allowed = {"development"} | ({"holdout"} if evaluate_holdout else set())
    return [item for item in load()["entries"] if item["split"] in allowed and item["smoke"]]


def run_smoke(entries: list[dict], mode: str) -> dict:
    report = {"generated_at": datetime.now().astimezone().isoformat(), "mode": mode, "rows": []}
    for entry in entries:
        command = [sys.executable, "-m", "bench.e2.worker", entry["id"]]
        started = time.monotonic()
        try:
            done = subprocess.run(command, cwd=PROJECT, capture_output=True, text=True, timeout=180, check=False)
        except subprocess.TimeoutExpired as error:
            row = {"id": entry["id"], "status": "Unknown", "reason": "worker_timeout", "stderr": str(error), "exit_code": None}
        else:
            try:
                detail = json.loads(done.stdout.splitlines()[-1])
            except (json.JSONDecodeError, IndexError):
                detail = {"id": entry["id"], "status": "Unknown", "reason": "invalid_worker_output", "stdout": done.stdout[-2000:]}
            row = {**detail, "stderr": done.stderr[-6000:], "exit_code": done.returncode}
            if done.returncode != 0 or detail.get("id") != entry["id"]:
                row["status"] = "failed" if detail.get("status") == "failed" else "Unknown"
        row["command"] = ["python", *command[1:]]
        row["elapsed_seconds"] = time.monotonic() - started
        report["rows"].append(row)
        (RESULTS / "e2_guard_smoke.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(entry["id"], row["status"], "exit=", row["exit_code"], flush=True)
    return report


def analyze(evaluate_holdout: bool = False, execute_smoke: bool = False) -> dict:
    entries = _entries(evaluate_holdout)
    mode = "development_and_frozen_holdout" if evaluate_holdout else "development_only"
    plans = {entry["id"]: build_plan(entry) for entry in entries}
    smoke_path = RESULTS / "e2_guard_smoke.json"
    smoke = run_smoke(entries, mode) if execute_smoke else json.loads(smoke_path.read_text(encoding="utf-8")) if smoke_path.exists() else {"mode": mode, "rows": []}
    smoke_rows = {row["id"]: row for row in smoke.get("rows", [])} if smoke.get("mode") == mode else {}
    rows = []
    for entry in entries:
        plan = plans[entry["id"]]
        smoke_row = smoke_rows.get(entry["id"])
        current = bool(smoke_row and smoke_row.get("plan_fingerprint") == plan.fingerprint)
        fast = bool(current and smoke_row.get("status") == "passed" and smoke_row.get("exit_code") == 0)
        rows.append(
            {
                "id": entry["id"],
                "source": entry["source"],
                "family": entry["family"],
                "split": entry["split"],
                "guard_status": plan.status,
                "guard_reason": plan.reason,
                "candidate_count": len(plan.candidates),
                "plan_fingerprint": plan.fingerprint,
                "smoke_current": current,
                "fast_feasible": fast,
                "smoke": smoke_row,
            }
        )
    guards = [row for row in rows if row["guard_status"] == "Supported"]
    fast_rows = [row for row in rows if row["fast_feasible"]]
    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "mode": mode,
        "holdout_used_for_rule_development": False,
        "counts": {"total": len(rows), "guard_available": len(guards), "fast_feasible": len(fast_rows)},
        "guard_by_family": dict(Counter(row["family"] for row in guards)),
        "fast_by_family": dict(Counter(row["family"] for row in fast_rows)),
        "guard_by_source": dict(Counter(row["source"] for row in guards)),
        "rows": rows,
        "scope": "Fast 可行仅表示冻结输入上 Guard 放行且数值参考通过；不授予在线 Fast，也不证明一般安全。",
    }


def write_report(data: dict) -> None:
    path = RESULTS / "e2_guard_coverage"
    path.with_suffix(".json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# e2 Guard 与 Fast 可行覆盖",
        "",
        f"- 模式：`{data['mode']}`。",
        f"- 计数：{data['counts']}。",
        f"- Guard 类别：{data['guard_by_family']}。",
        f"- Fast 类别：{data['fast_by_family']}。",
        f"- 边界：{data['scope']}",
        "",
        "| Kernel | 来源 | 类别 | Guard | Fast 可行 | 当前 smoke |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in data["rows"]:
        lines.append(f"| {row['id']} | {row['source']} | {row['family']} | {row['guard_status']} | {row['fast_feasible']} | {row['smoke_current']} |")
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate-holdout", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    data = analyze(args.evaluate_holdout, args.smoke)
    write_report(data)
    print(json.dumps(data["counts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
