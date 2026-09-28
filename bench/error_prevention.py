"""已知布局错读的原始 Kernel、旧 Guard 和新分派逐例隔离对照。"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from pact.oracle import run_isolated


ROOT = Path(__file__).resolve().parent.parent
CASES = (
    {"name": "A", "layout": "transpose", "m": 7, "n": 11},
    {"name": "A", "layout": "transpose", "m": 5, "n": 13},
    {"name": "D", "layout": "x_strided", "n": 129},
    {"name": "D", "layout": "y_strided", "n": 129},
)


def new_isolated(recipe: dict) -> dict:
    try:
        child = subprocess.run([sys.executable, "-m", "scenarios.guard_worker"], cwd=ROOT,
                               input=json.dumps(recipe), text=True, capture_output=True,
                               timeout=90, check=False)
    except subprocess.TimeoutExpired:
        return {"classification": "timeout_unknown", "exit_code": None}
    try:
        result = json.loads(child.stdout)
    except json.JSONDecodeError:
        result = {"classification": "process_failure_unknown", "stderr": child.stderr[-1000:]}
    result["exit_code"] = child.returncode
    return result


def build() -> dict:
    rows = []
    for recipe in CASES:
        kwargs = {"m": recipe.get("m", 7), "n": recipe.get("n", 11)}
        raw = run_isolated(recipe["name"], recipe["layout"], mode="raw", **kwargs).to_dict()
        old = run_isolated(recipe["name"], recipe["layout"], mode="dispatch", **kwargs).to_dict()
        new = new_isolated(recipe)
        prevented = (raw["category"] == "numeric_mismatch" and old["category"] == "correct"
                     and old["detail"].get("path") == "PyTorch Fallback"
                     and new["classification"] == "correct"
                     and new.get("detail", {}).get("path") == "PyTorch Fallback"
                     and new.get("detail", {}).get("direct_guard", {}).get("status") == "False")
        rows.append({"recipe": recipe, "raw": raw, "old": old, "new": new, "prevented": prevented})
    return {"generated_at": datetime.now().astimezone().isoformat(), "rows": rows,
            "prevented": sum(row["prevented"] for row in rows), "total": len(rows),
            "source_scope": "D 来自 Triton 官方向量加法教程的本地适配；违约风险是输入布局改变后的可复现模式，不声称上游教程原始 wrapper 存在历史 bug。",
            "limits": "只覆盖 A 和 D 的四个已知错读输入；异常、超时和未证实内存错误不能合并为已阻止错误。"}


def render(data: dict) -> str:
    lines = ["# 第六阶段受限布局违约拦截", "", f"生成时间：{data['generated_at']}",
             f"逐例阻止：{data['prevented']}/{data['total']}。", "",
             "| 输入 | 原始 Fast | 旧 Guard | 新分派 | 阻止且结果正确 |", "| --- | --- | --- | --- | --- |"]
    for row in data["rows"]:
        recipe = row["recipe"]
        lines.append(f"| {recipe['name']} {recipe['layout']} {recipe.get('m', 7)}×{recipe.get('n', 11)} | {row['raw']['category']} | {row['old']['category']} / {row['old']['detail'].get('path')} | {row['new']['classification']} / {row['new'].get('detail', {}).get('path')} | {row['prevented']} |")
    lines += ["", data["source_scope"], data["limits"], ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/phase6_error_prevention.md")
    args = parser.parse_args()
    data = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(data), encoding="utf-8")
    args.output.with_suffix(".json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"prevented={data['prevented']}/{data['total']}")
    return 0 if data["prevented"] == data["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
