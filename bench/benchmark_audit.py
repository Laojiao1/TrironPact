"""核对新增离线正向集的独立语义标签，并在独立 worker 中抽样物理边界。"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from bench.inventory import ROOT, build as build_inventory, function_source
from bench.specs import AUDIT_RECORDS, NEW_CASES, new_call
from pact.access_ir import Expr, parse_access_ir
from pact.call_site import audit_call_site


BENCH_IDS = tuple(NEW_CASES)


def _expr(op: str, *parts: Expr) -> Expr:
    return Expr(op, args=tuple(sorted(parts, key=Expr.key)) if op in {"add", "mul"} else parts)


def _param(ir, name: str) -> Expr:
    return Expr("param", ir.signature.index(name))


def _candidate(ir, name: str) -> tuple[bool, dict[str, tuple[int, ...]], str]:
    loads = [point for point in ir.accesses if point.kind == "load"]
    store = next((point for point in ir.accesses if point.kind == "store"), None)
    if not loads or store is None:
        return False, {}, "缺少 load/store"
    if name.startswith("bench_feature_"):
        lane, pid = Expr("lane", args=(_param(ir, "B"),)), Expr("pid", 0)
        expected = {
            "X": _expr("add", _expr("mul", pid, _param(ir, "S0")), lane),
            "F": lane,
            "OUT": _expr("add", _expr("mul", pid, _param(ir, "N")), lane),
        }
        mask = Expr("lt", args=(lane, _param(ir, "N")))
        actual = {point.tensor: point for point in ir.accesses}
        proven = set(actual) == set(expected) and all(actual[key].offset == value and actual[key].mask == mask for key, value in expected.items())
        conditions = {"X": (1,), "F": (0,)} if proven else {}
        return proven, conditions, "逐行地址绑定 X.stride(0)，X 列与特征向量均按单位地址广播读取"
    block = ir.signature[-1]
    linear = _expr("add", _expr("mul", Expr("pid", 0), _param(ir, block)), Expr("lane", args=(_param(ir, block),)))
    mask = Expr("lt", args=(linear, _param(ir, "N")))
    proven = all(point.offset == linear and point.mask == mask for point in ir.accesses)
    conditions = {point.tensor: (0,) for point in loads} if proven else {}
    return proven, conditions, "所有输入与连续输出使用同一受 mask 一维线性索引"


def run_workers(timeout: int = 90) -> list[dict]:
    rows = []
    for name in BENCH_IDS:
        try:
            done = subprocess.run([sys.executable, "-m", "scenarios.bench_worker", name], cwd=ROOT, capture_output=True, text=True, timeout=timeout, check=False)
            lines = [line for line in done.stdout.splitlines() if line.strip()]
            detail = json.loads(lines[-1]) if lines else {"kind": "missing_output"}
            rows.append({"kernel": name, "exit_code": done.returncode, "stderr": done.stderr[-2000:], "detail": detail})
        except subprocess.TimeoutExpired:
            rows.append({"kernel": name, "exit_code": None, "stderr": "", "detail": {"kind": "timeout_unknown"}})
    return rows


def summarize(workers: list[dict]) -> dict:
    inventory = build_inventory()
    catalog = {row["id"]: row for row in inventory["entries"]}
    static_contracts = {}
    for name in BENCH_IDS:
        item = catalog[name]
        source, _ = function_source(ROOT / item["file"], item["function"])
        ir = parse_access_ir(source, item["pointers"], {key: 4 for key in item["pointers"]})
        module = importlib.import_module(item["file"].removesuffix(".py").replace("/", "."))
        kernel = getattr(module, item["function"])
        wrapper = NEW_CASES[name][0]
        site = audit_call_site(wrapper, kernel.fn.__name__, new_call(name))
        proven, conditions, reason = _candidate(ir, name)
        proven = proven and site.status == "Supported"
        static_contracts[name] = {
            "status": "Statically-Proven" if proven else "Unknown",
            "condition_axes": conditions if proven else {},
            "reason": reason if proven else "; ".join(site.reasons) or "访存索引不能与受限语义对应",
            "call_site": {"status": site.status, "reasons": site.reasons, "source": new_call(name).source},
        }
    physical = []
    for worker in workers:
        contract = static_contracts[worker["kernel"]]
        for original in worker.get("detail", {}).get("rows", []):
            row = dict(original)
            holds = contract["status"] == "Statically-Proven" and all(
                all(row["inputs"][tensor]["shape"][axis] == 1 or row["inputs"][tensor]["stride"][axis] == 1 for axis in axes)
                for tensor, axes in contract["condition_axes"].items())
            row["candidate_decision"] = "True" if holds else ("False" if contract["status"] == "Statically-Proven" else "Unknown")
            row["kernel"] = worker["kernel"]
            physical.append(row)
    checks = {
        "six_distinct_function_bodies": len({catalog[name]["sha256"] for name in BENCH_IDS}) == 6,
        "all_ast_supported": all(catalog[name]["parse_status"] == "Supported" for name in BENCH_IDS),
        "complete_machine_readable_specs": set(AUDIT_RECORDS) == set(catalog) and all(
            all((record.wrapper, record.reference, record.dtype_domain, record.input_domain, record.allowed_paths, record.semantic_audit))
            and bool(record.parameter_bindings or record.exclusion_reason)
            and bool(record.core_contract or record.exclusion_reason) for record in AUDIT_RECORDS.values()),
        "call_bindings_audited": all(item["call_site"]["status"] == "Supported" for item in static_contracts.values()),
        "unit_stride_candidates_proven": all(item["status"] == "Statically-Proven" for item in static_contracts.values()),
        "workers_completed": len(workers) == 6 and all(row["exit_code"] == 0 for row in workers),
        "eligible_boundaries_correct": len([row for row in physical if row["truth"] == "eligible"]) == 12 and all(row["observation"] == "correct" for row in physical if row["truth"] == "eligible"),
        "ineligible_boundaries_witnessed": len([row for row in physical if row["truth"] == "ineligible"]) == 8 and all(row["observation"] == "numeric_mismatch" for row in physical if row["truth"] == "ineligible"),
        "shadow_candidates_match_labels": len(physical) == 20 and all((row["candidate_decision"] == "True") == (row["truth"] == "eligible") for row in physical),
    }
    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "checks": checks,
        "go_offline_benchmark_audit": all(checks.values()),
        "catalog_fingerprints": {name: catalog[name]["sha256"] for name in BENCH_IDS},
        "static_contracts": static_contracts,
        "physical_rows": physical,
        "workers": workers,
        "scope": "仅六个新增项目内离线 Kernel、固定一维与二维特征广播布局及本次 GPU 环境。标签来自独立逻辑索引、wrapper 调用绑定与实际 stride 审计；未编译 GuardPlan，也未扩大 Fast。",
    }


def render(data: dict) -> str:
    lines = ["# 第六阶段新增离线基准审计", "", f"生成时间：{data['generated_at']}", "", f"受限审计：{'通过' if data['go_offline_benchmark_audit'] else '未通过'}。", "", "| 检查 | 结果 |", "| --- | --- |"]
    lines += [f"| {name} | {'通过' if value else '未通过'} |" for name, value in data["checks"].items()]
    lines += ["", "| Kernel | 布局 | 独立标签 | 运行观察 | 不一致元素 |", "| --- | --- | --- | --- | --- |"]
    for worker in data["workers"]:
        for row in worker.get("detail", {}).get("rows", []):
            decision = next(item["candidate_decision"] for item in data["physical_rows"] if item["kernel"] == worker["kernel"] and item["layout"] == row["layout"])
            lines.append(f"| {worker['kernel']} | {row['layout']} | {row['truth']} / candidate={decision} | {row['observation']} | {row['mismatch_count']} |")
    lines += ["", data["scope"], ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/phase6_benchmark_audit.md")
    args = parser.parse_args()
    data = summarize(run_workers())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(data), encoding="utf-8")
    args.output.with_suffix(".json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"go_offline_benchmark_audit={data['go_offline_benchmark_audit']} workers={len(data['workers'])}")
    return 0 if data["go_offline_benchmark_audit"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
