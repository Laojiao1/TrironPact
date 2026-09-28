"""第六阶段报告完整性与研究目标缺口的机器核对。"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from bench.benchmark_audit import BENCH_IDS
from bench.inventory import ROOT, build as build_inventory
from pact.guard_plan import SPECS, compile_guard_plan


def _load(name: str) -> dict:
    path = ROOT / "results" / f"{name}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def phase6_source_fingerprint() -> str:
    paths = sorted((ROOT / "bench").glob("*.py"))
    paths += sorted((ROOT / "test").glob("test_phase6_*.py"))
    paths += [ROOT / "test/run_tests.py", ROOT / "scenarios/bench_kernels.py", ROOT / "scenarios/bench_worker.py", ROOT / "scenarios/external_vllm.py"]
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(str(path.relative_to(ROOT)).replace("\\", "/").encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def build() -> dict:
    inventory = build_inventory()
    stored_inventory = _load("phase6_bench_inventory")
    contract = _load("phase6_contract_eval")
    boundary = _load("phase6_boundary_ablation")
    errors = _load("phase6_error_prevention")
    perf = _load("phase6_dispatch_performance")
    guard = _load("phase6_guard_overhead")
    regression = _load("phase6_regression")
    benchmark_audit = _load("phase6_benchmark_audit")
    plans = {name: compile_guard_plan(name) for name in SPECS}
    counts = inventory["counts"]
    checks = {
        "baseline_tests_and_isolation": bool(regression.get("stage5_validation", {}).get("go_stage5_regression")),
        "phase6_regression_current": regression.get("stage6_source_fingerprint") == phase6_source_fingerprint(),
        "catalog_syntax_current": counts["registered"] == len(inventory["entries"])
        and stored_inventory.get("counts") == inventory["counts"]
        and json.dumps(stored_inventory.get("entries"), ensure_ascii=False, sort_keys=True) == json.dumps(inventory["entries"], ensure_ascii=False, sort_keys=True),
        "contract_no_false_fast_on_labeled_inputs": contract["metrics"]["counts"]["fp"] == 0 and contract["guard_report_go"],
        "benchmark_audit_current": benchmark_audit.get("catalog_fingerprints") == {name: next(row["sha256"] for row in inventory["entries"] if row["id"] == name) for name in BENCH_IDS},
        "contract_benchmark_audit_current": contract.get("benchmark_audit_generated_at") == benchmark_audit.get("generated_at"),
        "complete_benchmark_specs": benchmark_audit.get("checks", {}).get("complete_machine_readable_specs") is True and benchmark_audit.get("checks", {}).get("call_bindings_audited") is True,
        "benchmark_shadow_no_false_accept": benchmark_audit["go_offline_benchmark_audit"]
        and contract.get("benchmark_metrics", {}).get("counts", {}).get("total") == 20
        and all(contract["benchmark_metrics"]["counts"].get(key) == 0 for key in ("fp", "fn", "Unknown", "Unsupported")),
        "contract_plan_fingerprints_current": set(contract.get("plan_fingerprints", {})) == set(plans) and all(
            contract["plan_fingerprints"][name] == plan.fingerprint for name, plan in plans.items()),
        "error_prevention_limited": errors["prevented"] == errors["total"] == 4,
        "boundary_equal_worker_budget": len(boundary["arms"]) == 2 and len({arm["workers_used"] for arm in boundary["arms"]}) == 1,
        "performance_all_numerically_checked": len(perf["rows"]) == 13 and all(row["status"] == "measured" and row["same_semantics"] and row["fingerprint"] == plans[row["recipe"]["name"]].fingerprint for row in perf["rows"]),
        "guard_report_same_measurement": guard["source_generated_at"] == perf["generated_at"],
        "benchmark_15_to_20": 15 <= counts["registered"] <= 20,
        "at_least_three_semantic_families": set(counts["semantic_families"]) >= {"elementwise_mapping", "layout_2d", "feature_broadcast"},
        "at_least_two_supported_each_semantic_family": bool(counts["supported_positive_by_family"]) and all(value >= 2 for value in counts["supported_positive_by_family"].values()),
        "five_layer_coverage_reported": contract.get("catalog_coverage") == {key: counts[key] for key in ("registered", "syntax_supported", "syntax_unknown", "syntax_unsupported", "candidate_complete", "guard_available", "fast_feasible")}
        and (counts["registered"], counts["syntax_supported"], counts["candidate_complete"], counts["guard_available"], counts["fast_feasible"]) == (15, 14, 14, 8, 8),
        "two_pinned_external_sources": counts["pinned_external_sources"] >= 2,
        "real_smt_deletion_ablation": bool(boundary["smt_real_deletions"]),
    }
    required = ("baseline_tests_and_isolation", "phase6_regression_current", "catalog_syntax_current", "contract_no_false_fast_on_labeled_inputs",
                "benchmark_audit_current", "contract_benchmark_audit_current", "complete_benchmark_specs", "benchmark_shadow_no_false_accept",
                "contract_plan_fingerprints_current", "error_prevention_limited", "boundary_equal_worker_budget",
                "performance_all_numerically_checked", "guard_report_same_measurement", "benchmark_15_to_20",
                "at_least_three_semantic_families", "at_least_two_supported_each_semantic_family", "five_layer_coverage_reported", "two_pinned_external_sources")
    return {"generated_at": datetime.now().astimezone().isoformat(), "checks": checks,
            "required": required, "go_stage6": all(checks[key] for key in required),
            "non_go_reasons": [key for key in required if not checks[key]],
            "smt_note": "无真实可删除候选时记录零实际删除；该项属于研究证据缺口，不单独强制伪造正例。",
            "scope": f"机器检查核对当前 {counts['registered']} 个登记 Kernel 与阶段报告；challenge 的明确拒绝不冒充正向支持。不能代替独立真值审计、跨硬件复现或论文结论。"}


def render(data: dict) -> str:
    lines = ["# 第六阶段阶段状态核验", "", f"生成时间：{data['generated_at']}",
             f"Go/No-Go：{'Go' if data['go_stage6'] else 'No-Go（阶段目标未完成）'}。", "",
             "| 核验项 | 结果 |", "| --- | --- |"]
    lines += [f"| {key} | {'通过' if value else '未满足'} |" for key, value in data["checks"].items()]
    lines += ["", "未满足的必需项：" + ("、".join(data["non_go_reasons"]) or "无"),
              data["smt_note"], data["scope"], ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/phase6_stage_status.md")
    args = parser.parse_args()
    data = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(data), encoding="utf-8")
    args.output.with_suffix(".json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"go_stage6={data['go_stage6']} missing={len(data['non_go_reasons'])}")
    return 0 if data["go_stage6"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
