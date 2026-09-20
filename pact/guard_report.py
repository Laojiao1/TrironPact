"""第五阶段逐案例隔离 Guard/分派证据报告。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from pact.guard_plan import SPECS, compile_guard_plan


ROOT = Path(__file__).resolve().parent.parent
RECIPES = (
    {"name": "A", "layout": "contiguous"},
    {"name": "A", "layout": "transpose"},
    {"name": "A", "layout": "transpose", "policy": "prefer_repair"},
    {"name": "A", "layout": "single_row", "m": 1},
    {"name": "A", "layout": "single_col", "n": 1},
    {"name": "A2", "layout": "padded"},
    {"name": "A2", "layout": "transpose", "policy": "prefer_repair"},
    {"name": "B", "layout": "contiguous", "n": 127},
    {"name": "B", "layout": "contiguous", "n": 128},
    {"name": "B", "layout": "contiguous", "n": 129},
    {"name": "B", "layout": "offset", "n": 129},
    {"name": "B", "layout": "strided", "n": 129, "policy": "prefer_repair"},
    {"name": "D", "layout": "contiguous", "n": 129},
    {"name": "D", "layout": "x_offset", "n": 129},
    {"name": "D", "layout": "x_strided", "n": 129, "policy": "prefer_repair"},
    {"name": "D", "layout": "y_strided", "n": 129, "policy": "prefer_repair"},
    {"name": "holdout_vector", "layout": "strided", "n": 129},
    {"name": "holdout_matrix", "layout": "strided", "m": 7, "n": 11},
    {"name": "pointer_hint", "layout": "contiguous", "n": 128},
    {"name": "pointer_hint", "layout": "offset", "n": 128},
    {"name": "pointer_hint", "layout": "offset", "n": 128, "policy": "prefer_repair"},
    {"name": "index_hint", "layout": "offset", "n": 128},
)


def build_report() -> dict:
    records = []
    for recipe in RECIPES:
        child = subprocess.run([sys.executable, "-m", "scenarios.guard_worker"], cwd=ROOT,
                               input=json.dumps(recipe), text=True, capture_output=True, timeout=60, check=False)
        try:
            result = json.loads(child.stdout)
        except json.JSONDecodeError:
            result = {"classification": "worker_error", "stdout": child.stdout[-500:], "stderr": child.stderr[-500:]}
        records.append({"recipe": recipe, "exit_code": child.returncode, "result": result})
    plans = {name: compile_guard_plan(name).to_dict() for name in SPECS}
    checks = {
        "八个入口均有当前源码的受支持计划": len(plans) == 8 and all(plan["status"] == "Supported" for plan in plans.values()),
        "全部隔离样例数值正确": all(item["exit_code"] == 0 and item["result"]["classification"] == "correct" for item in records),
        "两个留出案例直接 Fast": all(next(item for item in records if item["recipe"]["name"] == name)["result"].get("detail", {}).get("path") == "Fast" for name in ("holdout_vector", "holdout_matrix")),
        "Fast、修复及 Fallback 均有实际路径": {item["result"].get("detail", {}).get("path") for item in records} >= {"Fast", "Relayout+Fast", "PyTorch Fallback"},
        "A2 padding 未复制": next(item for item in records if item["recipe"]["name"] == "A2" and item["recipe"]["layout"] == "padded")["result"].get("detail", {}).get("repair") is None,
        "指针提示错位拒绝且索引提示不要求指针对齐": next(item for item in records if item["recipe"]["name"] == "pointer_hint" and item["recipe"]["layout"] == "offset" and item["recipe"].get("policy") is None)["result"].get("detail", {}).get("path") == "PyTorch Fallback" and next(item for item in records if item["recipe"]["name"] == "index_hint")["result"].get("detail", {}).get("path") == "Fast",
    }
    return {"generated_at": datetime.now().astimezone().isoformat(), "phase": "第五阶段独立新分派",
            "plans": plans, "runs": records, "checks": checks, "go_guard": all(checks.values())}


def render(data: dict) -> str:
    lines = ["# 第五阶段 Guard 与分派隔离报告", "", f"生成时间：{data['generated_at']}",
             f"验收：{'通过' if data['go_guard'] else '未通过'}。", "", "## 检查", ""]
    lines += [f"- {'通过' if value else '失败'}：{key}" for key, value in data["checks"].items()]
    lines += ["", "## 逐例路径", "", "| 案例 | 布局 | 策略 | 结果 | 路径 | 直接 Guard | 修复输入 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for item in data["runs"]:
        recipe, result = item["recipe"], item["result"]
        detail = result.get("detail", {})
        repair = detail.get("repair")
        lines.append(f"| {recipe['name']} | {recipe['layout']} | {recipe.get('policy', 'cost')} | {result['classification']} | {detail.get('path', '—')} | {detail.get('direct_guard', {}).get('status', '—')} | {','.join(repair['copied']) if repair else '—'} |")
    lines += ["", "完整 JSON 含每条候选的用途、来源、证据、求值顺序与短路位置；有限隔离运行不替代支持域内的静态充分规则。", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/guard_dispatch_report.md")
    args = parser.parse_args()
    data = build_report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(data), encoding="utf-8")
    args.output.with_suffix(".json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"guard checks: {sum(data['checks'].values())}/{len(data['checks'])}")
    return 0 if data["go_guard"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
