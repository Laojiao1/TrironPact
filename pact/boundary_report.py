"""第四阶段离线边界证据、精化建议与 SMT 审计报告。"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from pact.candidate_report import build_candidate_report
from pact.candidates import Candidate
from pact.contract_dsl import Condition, ValueRef
from pact.mutation import MutationRecipe, TensorRecipe, shadow_candidates, standard_recipes
from pact.mutation_oracle import run_mutation
from pact.smt_refine import Domain, check_redundancy


ROOT = Path(__file__).resolve().parent.parent


def _case_candidates(report: dict, case: str) -> list[dict]:
    item = report["hints"][case] if case in ("pointer", "index") else report["cases"][case]
    return [candidate for group in item["candidates"].values() for candidate in group]


def _base_for(recipe: MutationRecipe) -> MutationRecipe:
    if len(recipe.inputs[0][1].shape) == 2:
        return next(item for item in standard_recipes() if item.case == recipe.case and item.label == "contiguous")
    if recipe.case == "D":
        return next(item for item in standard_recipes() if item.case == "D" and item.label == "contiguous")
    return next(item for item in standard_recipes() if item.case == recipe.case and item.label == "tile_exact")


def _shadow(recipe: MutationRecipe, candidates: list[dict], result: dict | None) -> list[dict]:
    ptrs = {name: meta["effective_ptr"] for name, meta in (result or {}).get("detail", {}).get("input_metadata", {}).items()}
    return shadow_candidates(recipe, candidates, ptrs=ptrs)


def _evidence_row(recipe: MutationRecipe, candidates: list[dict]) -> dict:
    base = _base_for(recipe)
    base_values = shadow_candidates(base, candidates)
    # worker 可在实际地址不满足提示时只输出元数据，不执行原始 Kernel。
    result = run_mutation(recipe)
    values = _shadow(recipe, candidates, result)
    changed = [index for index, (before, after) in enumerate(zip(base_values, values)) if before["value"] is not None and after["value"] is not None and before["value"] != after["value"]]
    category = result["category"]
    static_values = [item["value"] for item in values if item["evidence"] == "Statically-Proven"]
    if category in ("numeric_mismatch", "metadata_mismatch") and static_values and all(value is True for value in static_values):
        suggestion = "static_rule_conflict_review"
    elif category in ("numeric_mismatch", "metadata_mismatch") and changed:
        suggestion = "review_understrong_or_expected_rejection"
    elif category == "correct" and changed:
        suggestion = "review_overstrong_or_nonessential_condition"
    elif category == "correct":
        suggestion = "keep_observation_only"
    else:
        suggestion = "inconclusive"
    return {"id": recipe.id, "fingerprint": recipe.fingerprint, "recipe": recipe.to_dict(), "base_id": base.id, "target_family": recipe.target_family, "candidate_values": values, "base_values": base_values, "changed_candidates": changed, "mixed_mutation": len(changed) > 1, "category": category, "skip_reason": result["detail"].get("reason", "") if category == "preflight_skip" else "", "oracle": result, "suggestion": suggestion, "static_revision": False}


def _span_shadow_boundary(report: dict) -> dict:
    # 仅在元数据上测试刚好到 storage 末端与差一格；不构造非法 GPU 视图。
    original = next(item for item in report["cases"]["B"]["candidates"]["span"] if item["origin"]["access_kind"] == "load")
    candidate = Candidate.from_dict(original)
    from pact.contract_dsl import TensorMetadata

    exact = TensorMetadata((129,), (1,), 1, 0x1004, 4, 520)
    short = TensorMetadata((129,), (1,), 1, 0x1004, 4, 516)
    output = TensorMetadata((129,), (1,), 0, 0x2000, 4, 516)
    return {"candidate_origin": asdict(candidate.origin), "exact_last_element": candidate.evaluate({"X": exact, "OUT": output}, {"N": 129, "B": 128}), "one_past_storage": candidate.evaluate({"X": short, "OUT": output}, {"N": 129, "B": 128}), "gpu_execution": False}


def _smt_audit(report: dict) -> dict:
    # 第三阶段没有两条真正冗余的同目的/同域条件。用与真实指针对齐候选同形的
    # 受限 fixture 验证删除逻辑；真实 Unproven 候选绝不提升为静态证明。
    pointer = Candidate.from_dict(report["hints"]["pointer"]["candidates"]["alignment"][0])
    strong = Candidate(pointer.condition, None, pointer.purpose, "Statically-Proven", pointer.origin, pointer.rule, pointer.domain, pointer.binding, "SMT 合成蕴含 fixture；不属于真实候选")
    weak = Candidate(Condition("mod_eq", left=ValueRef("effective_ptr", tensor="X"), divisor=4, remainder=0), None, pointer.purpose, "Statically-Proven", pointer.origin, pointer.rule, pointer.domain, pointer.binding, "SMT 合成蕴含 fixture；不属于真实候选")
    domain = Domain((("X", 1),))
    fixture = check_redundancy([strong, weak], domain)
    unproven = check_redundancy([pointer, weak], domain)
    return {"solver": "Z3", "fixture_only": True, "fixture": fixture, "actual_unproven_kept": unproven["kept"] == [0, 1] and unproven["removed"] == [], "actual_candidate_deletions": [], "reason": "第三阶段真实候选没有经同域完整形式化可删除的冗余对；指针条件仍为 Unproven"}


def _legacy_singleton(regression: dict) -> dict:
    """复核已存在的 PoC 见证，不把它伪装成第四阶段新发现。"""
    records = [row for row in regression.get("runs", ()) if row.get("detail", {}).get("case") == "A" and row.get("detail", {}).get("layout") in {"single_row", "single_col"}]
    checks = []
    for layout in ("single_row", "single_col"):
        subset = [row for row in records if row["detail"]["layout"] == layout]
        raw = any(row["category"] == "correct" and row["detail"]["mode"] == "raw" for row in subset)
        rejected = any(row["category"] == "correct" and row["detail"]["mode"] == "dispatch" and not row["detail"]["refined_contract"] and row["detail"]["path"] == "PyTorch Fallback" for row in subset)
        revised = any(row["category"] == "correct" and row["detail"]["mode"] == "dispatch" and row["detail"]["refined_contract"] and row["detail"]["path"] == "Fast" for row in subset)
        checks.append({"layout": layout, "raw_correct": raw, "old_rejected": rejected, "static_revision_verified": revised})
    return {"source": "第三阶段候选回归中保留的第一阶段 A singleton 见证", "checks": checks, "static_reason": regression.get("refinement", {}).get("static_reason", ""), "verified": bool(regression.get("refinement_verified")) and all(all(value for key, value in item.items() if key != "layout") for item in checks)}


def _legacy_path(recipe: MutationRecipe, regression: dict) -> str:
    if recipe.case in ("holdout_vector", "holdout_matrix"):
        return "Unsupported"
    specs = dict(recipe.inputs)
    spec = specs["X"]
    for run in regression.get("runs", ()):
        detail = run.get("detail", {})
        if detail.get("case") != recipe.case or detail.get("mode") != "dispatch" or tuple(detail.get("shape", ())) != spec.shape or tuple(detail.get("stride", ())) != spec.stride or detail.get("storage_offset") != spec.offset:
            continue
        if "Y" in specs and (tuple(detail.get("second_stride") or ()) != specs["Y"].stride or specs["Y"].offset != 0):
            # 旧报告未保存第二输入的 offset；不能在未知时声称路径相同。
            continue
        return detail.get("path", "Unknown")
    return "NotObserved"


def build_boundary_report(*, quick: bool = False) -> dict:
    regression_path = ROOT / "results" / "candidate_regression.json"
    regression = json.loads(regression_path.read_text(encoding="utf-8"))
    candidates = build_candidate_report(regression)
    recipes = standard_recipes()
    if quick:
        preferred = {"A": "transpose", "A2": "padded", "B": "tile_plus", "D": "x_stride", "holdout_vector": "stride_two", "holdout_matrix": "transpose", "pointer": "tile_exact", "index": "offset"}
        recipes = tuple(item for item in recipes if preferred.get(item.case) == item.label)
    rows = [_evidence_row(recipe, _case_candidates(candidates, recipe.case)) for recipe in recipes]
    for row in rows:
        row["legacy_guard_path"] = _legacy_path(MutationRecipe.from_dict(row["recipe"]), regression)
    counts = {kind: sum(row["category"] == kind for row in rows) for kind in sorted({row["category"] for row in rows})}
    span = _span_shadow_boundary(candidates)
    smt = _smt_audit(candidates)
    legacy = _legacy_singleton(regression)
    checks = {
        "变异覆盖八个案例且每例独立隔离": {row["recipe"]["case"] for row in rows} == {"A", "A2", "B", "D", "holdout_vector", "holdout_matrix", "pointer", "index"} and all(row["oracle"] is None or row["oracle"]["detail"].get("fingerprint") == row["fingerprint"] for row in rows),
        "执行分类完整且无未解释故障": all(row["category"] in {"correct", "numeric_mismatch", "metadata_mismatch", "preflight_skip"} for row in rows),
        "已知错读与正确边界分开": any(row["category"] == "numeric_mismatch" and row["recipe"]["case"] == "A" for row in rows) and any(row["category"] == "correct" and row["recipe"]["case"] == "A2" for row in rows),
        "双输入旧路径同时核对 X 与 Y": all(row["legacy_guard_path"] == "PyTorch Fallback" for row in rows if row["recipe"]["case"] == "D" and row["recipe"]["label"] in {"x_stride", "y_stride"}),
        "指针提示不满足时不强制运行": quick or any(row["recipe"]["case"] == "pointer" and row["category"] == "preflight_skip" for row in rows),
        "末端与越界一格只作元数据影子": span["exact_last_element"] is True and span["one_past_storage"] is False and not span["gpu_execution"],
        "SMT 冗余 fixture 不升级真实义务": smt["fixture"]["removed"] == [1] and smt["actual_unproven_kept"] and not smt["actual_candidate_deletions"],
        "历史过强见证具独立静态修订理由": legacy["verified"] and bool(legacy["static_reason"]),
        "静态候选没有未处理的满足集错读": not any(row["suggestion"] == "static_rule_conflict_review" for row in rows),
        "第三阶段基线仍通过": regression.get("go_core") is True,
    }
    return {"generated_at": datetime.now().astimezone().isoformat(), "quick": quick, "phase": "第四阶段离线证据与影子精化；未接入 Guard", "rows": rows, "counts": counts, "span_shadow": span, "smt": smt, "legacy_singleton": legacy, "checks": checks, "go_boundary": all(checks.values())}


def render_boundary_report(report: dict) -> str:
    lines = ["# TritonPact 第四阶段边界证据与精化报告", "", f"> 生成时间：{report['generated_at']}", f"> 验收：{'通过' if report['go_boundary'] else '未通过'}；{'快速' if report['quick'] else '完整'}运行", "> 新证据与精化建议只作离线/影子分析，旧 Guard 未接入。", "", "## 检查", "", "| 检查项 | 结果 |", "| --- | --- |"]
    lines.extend(f"| {name} | {'通过' if passed else '失败'} |" for name, passed in report["checks"].items())
    lines.extend(["", "## 逐例变异", "", "| ID | 目标 | 候选变化 | 混合 | Oracle | 旧 Guard 对照 | 建议 |", "| --- | --- | --- | --- | --- | --- | --- |"])
    for row in report["rows"]:
        lines.append(f"| `{row['id']}` | {row['target_family']} | {row['changed_candidates']} | {'是' if row['mixed_mutation'] else '否'} | {row['category']} | {row['legacy_guard_path']} | {row['suggestion']} |")
    lines.extend(["", "## 计数与逻辑边界", "", f"- 分类计数：`{json.dumps(report['counts'], ensure_ascii=False, sort_keys=True)}`。", f"- Span 末端：{report['span_shadow']['exact_last_element']}；越界一格：{report['span_shadow']['one_past_storage']}；仅元数据影子，无非法 GPU 视图运行。", f"- 历史 A singleton 过强见证：{report['legacy_singleton']['verified']}；静态理由：{report['legacy_singleton']['static_reason']}。", f"- SMT 合成冗余 fixture 删除索引：{report['smt']['fixture']['removed']}；真实候选删除：{report['smt']['actual_candidate_deletions']}。", "- 正确补集与错读只构成所测输入的证据；混合变异不作单谓词因果归因。Z3 的合成 fixture 不提升真实动态指针对齐义务。", "- 完整输入配方、实际地址余数、参考比较、子进程诊断、逐候选影子值及 SMT 查询见同名 JSON。", ""])
    return "\n".join(lines)
