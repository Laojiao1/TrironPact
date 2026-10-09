"""生成 e5a 的 CCF B readiness 与 CCF A gap 机器状态。"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

from bench.e1.audit import fingerprint as e1_fingerprint
from bench.e2.freeze import current_fingerprint
from bench.e3.validate import source_fingerprint as e3_source_fingerprint
from bench.e4.validate import source_fingerprint as e4_source_fingerprint


PROJECT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT / "results"


def _load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT, capture_output=True, text=True, timeout=30, check=True).stdout.rstrip("\r\n")


def _second_review_complete(risk: dict) -> bool:
    """只有结构完整的独立审阅记录才满足最终条件；文件缺失即 False。"""

    path = RESULTS / "e5_second_review.json"
    if not path.is_file():
        return False
    try:
        review = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return False
    risk_ids = {
        row["case_id"] for row in risk.get("rows", ()) if row.get("status") == "complete"
    }
    return (
        review.get("reviewer_independent") is True
        and review.get("reviewed_kernel_count", 0) >= 7
        and review.get("formal_kernel_denominator") == 28
        and set(review.get("reviewed_risk_case_ids", ())) == risk_ids
        and review.get("disagreements_resolved") is True
        and bool(review.get("review_records"))
    )


def build_ccfb(*, verification_context: dict | None = None) -> dict:
    e1 = _load("e1_semantic_audit")
    e2 = _load("e2_regression")
    freeze = json.loads((PROJECT / "bench/e2/freeze.json").read_text(encoding="utf-8"))
    e3 = _load("e3_regression")
    risk = _load("e3_real_risk_cases")
    mutation = _load("e3_mutation_comparison")
    smt = _load("e3_smt_audit")
    e4 = _load("e4_regression")
    correctness = _load("e4_workload_correctness")
    baselines = _load("e4_dispatch_baselines")
    diagnostics = _load("e4_wsl_diagnostics")
    cross = _load("e4_cross_hardware_performance")
    holdout = _load("e5_holdout_evaluation")
    trace = _load("e5_traceability")
    manifest = _load("e5_evidence_manifest")
    context = verification_context or {}
    initial_clean = context.get("initial_worktree_clean", _git("status", "--short") == "")
    one_click_passed = context.get("one_click_checks_passed", False)
    current_head = _git("rev-parse", "HEAD")
    conditions = {
        "1_real_corpus_coverage_and_holdout_thresholds": e1.get("fingerprints") == e1_fingerprint()
        and e2.get("go_e2") is True
        and freeze.get("rule_fingerprint") == current_fingerprint()
        and holdout.get("go_holdout_audit") is True,
        "2_reproducible_risk_evidence": e3.get("go_e3") is True
        and e3.get("source_fingerprint") == e3_source_fingerprint()
        and risk.get("go_risk") is True
        and risk["counts"]["l1_witnesses"] >= 3
        and len(risk["counts"]["by_source"]) >= 2,
        "3_zero_known_holdout_false_allows": holdout["counts"]["known_false_allows"] == 0
        and holdout["checks"]["unknown_not_fast"],
        "4_two_workloads_and_two_hardware_environments": e4.get("go_e4_integration") is True
        and e4.get("source_fingerprint") == e4_source_fingerprint()
        and cross.get("go_e4_cross_hardware") is True
        and cross.get("obtained_environments", 0) >= 2,
        "5_fair_system_baselines": baselines.get("go_baselines") is True and all(baselines.get("checks", {}).values()),
        "6_correctness_argument_traceable": (PROJECT / "artifact/e5_correctness.md").is_file()
        and trace.get("go_traceability") is True,
        "7_independent_second_review": _second_review_complete(risk),
        "8_secondary_and_zero_results_disclosed": mutation.get("go_mutation") is True
        and smt.get("go_smt") is True
        and smt["counts"]["real_candidate_deletions"] == 1
        and smt["counts"]["synthetic_deletions"] == 0
        and diagnostics.get("checks", {}).get("diagnostic_scope_only") is True,
        "9_current_commit_environment_and_data_bound": initial_clean
        and manifest.get("git_head") == current_head
        and all(manifest.get("checks", {}).values()),
        "10_clean_checkout_one_click_verification": initial_clean
        and one_click_passed
        and current_head == _git("rev-parse", "origin/eval"),
    }
    missing = {
        "4_two_workloads_and_two_hardware_environments": "e4b 尚无两套原生 Linux GPU、实际 smoke、正确性和开销复核。",
        "7_independent_second_review": "尚无第二审阅人对至少 25% 正式 Kernel 及全部风险案例的独立记录。",
        "9_current_commit_environment_and_data_bound": "e5a 尚在未提交开发工作树；提交后需在目标提交重新生成最终证据清单。",
        "10_clean_checkout_one_click_verification": "最终条件要求提交后的干净 checkout 重跑；e5a 开发验收不能提前满足。",
    }
    rows = [
        {"id": key, "passed": value, "reason": "证据满足" if value else missing.get(key, "证据缺失或失配")}
        for key, value in conditions.items()
    ]
    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "ready" if all(conditions.values()) else "not_ready",
        "go_e5_ccfb_readiness": all(conditions.values()),
        "go_e5": all(conditions.values()),
        "conditions": conditions,
        "rows": rows,
        "missing": [row for row in rows if not row["passed"]],
        "scope": "这些是项目路线的内部论文证据门槛，不是 CCF 会议官方录用规则。",
        "observed": {"workloads": correctness["counts"]["workloads_complete"], "known_false_allows": correctness["counts"]["known_false_allows"]},
    }


def build_ccfa() -> dict:
    gaps = [
        {"id": "unified_formal_model", "status": "open", "evidence": "当前为支持子语言和证明草图，尚无机器可检证明或更完整形式化模型。"},
        {"id": "confirmed_upstream_impact", "status": "open", "evidence": "尚无由本项目新发现并获维护者确认的未知缺陷或上游合入。"},
        {"id": "compiler_graph_integration", "status": "open", "evidence": "已有真实 Inductor 图执行，但没有图级 Guard 或编译器级部署收益。"},
        {"id": "guided_mutation_advantage", "status": "open", "evidence": "冻结实验不支持谓词引导普遍优于强随机基线。"},
        {"id": "cross_generation_results", "status": "open", "evidence": "原生 Linux 双硬件尚未完成，更无多 GPU 代际一致结论。"},
    ]
    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "gap_recorded",
        "go_e5_ccfa": False,
        "gaps": gaps,
        "policy": "CCF A 不设置数量型 Go；需在 CCF B 证据完整后依据同期工作重新制定目标。",
    }


def write(ccfb: dict, ccfa: dict) -> None:
    (RESULTS / "e5_ccfb_readiness.json").write_text(json.dumps(ccfb, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# e5 CCF B readiness",
        "",
        f"- 状态：`{ccfb['status']}`；`go_e5_ccfb_readiness={str(ccfb['go_e5_ccfb_readiness']).lower()}`。",
        f"- {ccfb['scope']}",
        "",
        "| 最终条件 | 结果 | 原因 |",
        "| --- | --- | --- |",
        *[f"| `{row['id']}` | {'通过' if row['passed'] else '未满足'} | {row['reason']} |" for row in ccfb["rows"]],
        "",
    ]
    (RESULTS / "e5_ccfb_readiness.md").write_text("\n".join(lines), encoding="utf-8")
    (RESULTS / "e5_ccfa_gap.json").write_text(json.dumps(ccfa, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    gap_lines = [
        "# e5 CCF A gap",
        "",
        f"- 状态：`{ccfa['status']}`；`go_e5_ccfa=false`。",
        f"- {ccfa['policy']}",
        "",
        *[f"- **{row['id']}**（{row['status']}）：{row['evidence']}" for row in ccfa["gaps"]],
        "",
    ]
    (RESULTS / "e5_ccfa_gap.md").write_text("\n".join(gap_lines), encoding="utf-8")


def main() -> int:
    argparse.ArgumentParser().parse_args()
    ccfb, ccfa = build_ccfb(), build_ccfa()
    write(ccfb, ccfa)
    print(json.dumps({"go_e5_ccfb_readiness": ccfb["go_e5_ccfb_readiness"], "go_e5_ccfa": ccfa["go_e5_ccfa"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

