"""在既有受限案例上重新隔离执行，核对 Guard 与独立审计标签。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from bench.inventory import build as build_inventory
from bench.metrics import decision_metrics
from pact.guard_report import RECIPES, ROOT, build_report


# 标签来自受限索引/绑定审计和已知指针提示义务；不从本次 Guard 输出反推。
# 状态仅对表中确切布局、shape、dtype 和当前源码成立。
LABELS = {
    ("A", "contiguous", 7, 11): ("eligible", "A: 二维线性地址与连续布局对应"),
    ("A", "transpose", 7, 11): ("ineligible", "A: 转置后线性地址与 x[row,col] 不对应"),
    ("A", "single_row", 1, 11): ("eligible", "A: row 恒为 0，行步长无作用"),
    ("A", "single_col", 7, 1): ("eligible", "A: col 恒为 0，列步长无作用"),
    ("A2", "padded", 7, 11): ("eligible", "A2: 行步长已绑定，列步长为 1"),
    ("A2", "transpose", 7, 11): ("ineligible", "A2: 列步长不是 1"),
    ("B", "contiguous", 7, 127): ("eligible", "B: 单位步长，尾部 mask 覆盖 N"),
    ("B", "contiguous", 7, 128): ("eligible", "B: 单位步长，整 Tile"),
    ("B", "contiguous", 7, 129): ("eligible", "B: 单位步长，尾部 mask 覆盖 N"),
    ("B", "offset", 7, 129): ("eligible", "B: 有效首地址偏移不改变单位逻辑步长"),
    ("B", "strided", 7, 129): ("ineligible", "B: 原始 idx 地址与步长视图的逻辑 idx 不对应"),
    ("D", "contiguous", 7, 129): ("eligible", "D: X/Y 均单位步长且 mask 覆盖 N"),
    ("D", "x_offset", 7, 129): ("eligible", "D: X 有 offset 但单位步长"),
    ("D", "x_strided", 7, 129): ("ineligible", "D: X 非单位步长发生错读"),
    ("D", "y_strided", 7, 129): ("ineligible", "D: Y 非单位步长发生错读"),
    ("holdout_vector", "strided", 7, 129): ("eligible", "显式 S=x.stride(0) 的逐元素读取"),
    ("holdout_matrix", "strided", 7, 11): ("eligible", "显式 S0/S1 与当前输入步长绑定"),
    ("pointer_hint", "contiguous", 7, 128): ("eligible", "有效指针满足 16 字节提示时本次可进入 Fast"),
    ("pointer_hint", "offset", 7, 128): ("ineligible", "4 字节 offset 后有效指针不满足 16 字节提示"),
    ("index_hint", "offset", 7, 128): ("eligible", "提示约束 idx 的倍数，不约束 X 指针"),
}


def recipe_key(recipe: dict) -> tuple:
    return recipe["name"], recipe["layout"], recipe.get("m", 7), recipe.get("n", 11 if recipe["name"] in {"A", "A2", "holdout_matrix"} else (128 if recipe["name"] in {"pointer_hint", "index_hint"} else 129))


def summarize(report: dict) -> dict:
    rows = []
    seen = set()
    for item in report["runs"]:
        recipe = item["recipe"]
        key = recipe_key(recipe)
        if key in seen:
            continue  # 同一物理输入的不同 repair_policy 不重复计分。
        seen.add(key)
        truth, source = LABELS.get(key, ("unlabeled", ""))
        result = item["result"]
        direct = result.get("detail", {}).get("direct_guard", {}).get("status")
        decision = direct if direct in {"True", "False", "Unknown", "Unsupported"} else "Unknown"
        rows.append({"id": ":".join(map(str, key)), "kernel": key[0], "recipe": recipe, "evidence_mode": "online_guard",
                     "truth": truth, "truth_source": source, "decision": decision,
                     "classification": result.get("classification"), "path": result.get("detail", {}).get("path"),
                     "worker_exit_code": item.get("exit_code")})
    benchmark_path = ROOT / "results" / "phase6_benchmark_audit.json"
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8")) if benchmark_path.is_file() else {"physical_rows": []}
    benchmark_rows = [{"id": f"{item['kernel']}:{item['layout']}", "kernel": item["kernel"], "recipe": {"layout": item["layout"]},
                       "evidence_mode": "offline_shadow_candidate", "truth": item["truth"], "truth_source": "独立一维索引与物理 stride 审计",
                       "decision": item["candidate_decision"], "classification": item["observation"], "path": "Offline Shadow"}
                      for item in benchmark.get("physical_rows", [])]
    metric = decision_metrics(rows)
    benchmark_metric = decision_metrics(benchmark_rows)
    combined = decision_metrics(rows + benchmark_rows)
    inventory = build_inventory()
    return {"generated_at": datetime.now().astimezone().isoformat(), "mode": "fresh_isolated_limited_domain",
            "guard_report_checks": report["checks"], "guard_report_go": report["go_guard"],
            "plan_fingerprints": {name: item["fingerprint"] for name, item in report.get("plans", {}).items()},
            "rows": rows, "benchmark_shadow_rows": benchmark_rows,
            "benchmark_audit_generated_at": benchmark.get("generated_at"), "metrics": metric,
            "benchmark_metrics": benchmark_metric, "combined_metrics": combined,
            "catalog_coverage": {key: inventory["counts"][key] for key in ("registered", "syntax_supported", "syntax_unknown", "syntax_unsupported", "candidate_complete", "guard_available", "fast_feasible")},
            "label_scope": "仅既有确切输入；标签来自索引/调用/提示义务的人工审计，未经第二名审计者盲审。不能外推契约恢复总体 precision/recall。"}


def render(data: dict) -> str:
    m = data["metrics"]
    c = m["counts"]
    lines = ["# 第六阶段受限契约判定试运行", "", f"生成时间：{data['generated_at']}",
             "", f"隔离报告通过：{data['guard_report_go']}；独立标签 {c['labeled']}，明确判定 {c['decided']}，Unknown {c['Unknown']}，Unsupported {c['Unsupported']}。",
             f"TP={c['tp']} FP={c['fp']} TN={c['tn']} FN={c['fn']}；precision={m['precision']}，已判定子集 recall={m['recall_on_decided']}。",
             f"按 Kernel 宏平均 precision={m['macro_by_kernel']['precision']['value']}（{m['macro_by_kernel']['precision']['defined_kernels']} 个可定义分母），recall={m['macro_by_kernel']['recall_on_decided']['value']}（{m['macro_by_kernel']['recall_on_decided']['defined_kernels']} 个可定义分母）。",
             f"离线影子候选单列：标签 {data['benchmark_metrics']['counts']['labeled']}，FP={data['benchmark_metrics']['counts']['fp']}，FN={data['benchmark_metrics']['counts']['fn']}，Unknown={data['benchmark_metrics']['counts']['Unknown']}。",
             f"合并仅用于样例总览：标签 {data['combined_metrics']['counts']['labeled']}，FP={data['combined_metrics']['counts']['fp']}，FN={data['combined_metrics']['counts']['fn']}，Unknown={data['combined_metrics']['counts']['Unknown']}。",
             f"全清单五层覆盖：{data['catalog_coverage']}。Unsupported 进入清单分母，不进入有标签输入 precision/recall。",
             "", "| 输入 | 模式 | 独立标签 | 判定 | 路径 | 运行分类 |", "| --- | --- | --- | --- | --- | --- |"]
    for row in data["rows"] + data.get("benchmark_shadow_rows", []):
        lines.append(f"| {row['id']} | {row.get('evidence_mode', 'online_guard')} | {row['truth']} | {row['decision']} | {row['path']} | {row['classification']} |")
    lines += ["", data["label_scope"], m["meaning"], ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/phase6_contract_eval.md")
    parser.add_argument("--rerender-existing", action="store_true", help="只重算同名 JSON 的统计值，不再次运行隔离 worker")
    args = parser.parse_args()
    if args.rerender_existing:
        data = json.loads(args.output.with_suffix(".json").read_text(encoding="utf-8"))
        data["metrics"] = decision_metrics(data["rows"])
        data["benchmark_metrics"] = decision_metrics(data.get("benchmark_shadow_rows", []))
        data["combined_metrics"] = decision_metrics(data["rows"] + data.get("benchmark_shadow_rows", []))
        inventory = build_inventory()
        data["catalog_coverage"] = {key: inventory["counts"][key] for key in ("registered", "syntax_supported", "syntax_unknown", "syntax_unsupported", "candidate_complete", "guard_available", "fast_feasible")}
    else:
        if set(recipe_key(item) for item in RECIPES) != set(LABELS):
            raise ValueError("现有配方与独立标签集合不一致，必须重新审计")
        data = summarize(build_report())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(data), encoding="utf-8")
    args.output.with_suffix(".json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"labeled={data['metrics']['counts']['labeled']} fp={data['metrics']['counts']['fp']} unknown={data['metrics']['counts']['Unknown']}")
    return 0 if data["guard_report_go"] and data["metrics"]["counts"]["fp"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
