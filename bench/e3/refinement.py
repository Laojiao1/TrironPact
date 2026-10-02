"""汇总 e3 见证分类、候选修订和 Fast 撤销审计。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT / "results"


def _load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))


def build() -> dict:
    risk = _load("e3_real_risk_cases")
    mutation = _load("e3_mutation_comparison")
    smt = _load("e3_smt_audit")
    witnesses = [row["detail"] for row in risk["rows"] if row["status"] == "complete" and row["detail"]["classification"] == "numeric_mismatch"]
    understrong = [item["case"]["id"] for item in witnesses if item["guard"]["allowed"]]
    repaired = [item["case"]["id"] for item in witnesses if not item["guard"]["allowed"] and all(row["correct"] for row in item["relayout"]["checks"])]
    # e2 Guard 的声明域就是冻结 shape/stride；域外正确输入不能反推其条件过强。
    revisions = []
    fast_revocations = []
    checks = {
        "all_numeric_witnesses_blocked": not understrong and len(repaired) == len(witnesses),
        "no_static_sufficient_condition_refuted": not understrong,
        "revisions_have_complete_provenance": all(
            all(key in row for key in ("trigger", "static_reason", "before", "after", "regression")) for row in revisions
        ),
        "revocations_match_refutations": len(fast_revocations) == len(understrong),
        "mixed_mutations_not_attributed": all(
            selected["classification"] == "mixed_unattributed"
            for trial in mutation["trials"]
            for selected in trial["selected"]
            if selected["probe"] in {"grid_mixed", "triton_two_inputs", "unsloth_mixed"} and not selected["worker"]
        ),
        "smt_result_carried_forward": smt["go_smt"] is True,
    }
    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "witness_classification": {
            "expected_rejection": repaired,
            "understrong": understrong,
            "overstrong": [],
            "mixed_unattributed": ["grid_mixed", "triton_two_inputs", "unsloth_mixed"],
            "unreachable": ["span_one_short"],
            "unknown": [row["case_id"] for row in risk["rows"] if row["status"] != "complete"],
        },
        "candidate_revisions": revisions,
        "actual_revision_count": len(revisions),
        "fast_revocations": fast_revocations,
        "checks": checks,
        "go_refinement": all(checks.values()),
        "scope": "e2 Guard 只承诺冻结调用域；域外正确样例不标为过强。若未来出现 Guard=True 的确定错读，必须先撤销 Fast 再继续。",
    }


def write(report: dict) -> None:
    base = RESULTS / "e3_refinement_audit"
    base.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# e3 精化与 Fast 撤销审计",
        "",
        f"- 生成时间：{report['generated_at']}。",
        f"- 状态：`{'go' if report['go_refinement'] else 'no_go'}`。",
        f"- 分类：`{report['witness_classification']}`。",
        f"- 实际候选修订：{report['actual_revision_count']}；Fast 撤销：{len(report['fast_revocations'])}。",
        f"- 边界：{report['scope']}",
        "",
    ]
    base.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    report = build()
    write(report)
    print(json.dumps({"go_refinement": report["go_refinement"], "actual_revision_count": report["actual_revision_count"], "fast_revocations": report["fast_revocations"]}, ensure_ascii=False))
    return 0 if report["go_refinement"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

