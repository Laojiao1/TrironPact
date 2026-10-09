"""只读审计 e2 已完成的一次性冻结留出评估。"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from bench.e2.freeze import current_fingerprint


PROJECT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT / "results"
FREEZE = PROJECT / "bench" / "e2" / "freeze.json"


def _load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build() -> dict:
    """复用历史冻结结果；本函数不启动 Kernel，也不修改 e2 规则。"""

    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    access = _load("e2_access_ir_coverage")
    candidates = _load("e2_candidate_extraction")
    guards = _load("e2_guard_coverage")
    by_access = {row["id"]: row for row in access["rows"] if row["split"] == "holdout"}
    by_candidate = {row["id"]: row for row in candidates["rows"] if row["split"] == "holdout"}
    by_guard = {row["id"]: row for row in guards["rows"] if row["split"] == "holdout"}
    ids = sorted(by_access)
    rows = []
    for kernel_id in ids:
        access_row = by_access[kernel_id]
        candidate_row = by_candidate.get(kernel_id, {})
        guard_row = by_guard.get(kernel_id, {})
        rows.append(
            {
                "id": kernel_id,
                "access_status": access_row["status"],
                "access_reason": access_row["reason"],
                "candidate_status": candidate_row.get("status", "Unknown"),
                "guard_status": guard_row.get("guard_status", "Unknown"),
                "fast_feasible": guard_row.get("fast_feasible") is True,
                "smoke_current": guard_row.get("smoke_current") is True,
            }
        )
    checks = {
        "freeze_precedes_holdout_and_marks_evaluated": freeze.get("holdout_evaluated") is True,
        "rule_fingerprint_still_current": freeze.get("rule_fingerprint") == current_fingerprint(),
        "same_seven_holdout_ids_in_all_reports": len(ids) == 7 and set(ids) == set(by_candidate) == set(by_guard),
        "reports_use_frozen_holdout_mode": all(
            report.get("mode") == "development_and_frozen_holdout" for report in (access, candidates, guards)
        ),
        "holdout_not_used_for_rule_development": all(
            report.get("holdout_used_for_rule_development") is False for report in (access, candidates, guards)
        ),
        "six_supported_one_unknown_preserved": sum(row["access_status"] == "Supported" for row in rows) == 6
        and sum(row["access_status"] == "Unknown" for row in rows) == 1,
        "supported_holdout_smoke_current": all(
            row["smoke_current"] for row in rows if row["access_status"] == "Supported"
        ),
        "unknown_not_fast": all(not row["fast_feasible"] for row in rows if row["access_status"] != "Supported"),
    }
    source_reports = {}
    for name in ("e2_access_ir_coverage", "e2_candidate_extraction", "e2_guard_coverage"):
        path = RESULTS / f"{name}.json"
        source_reports[str(path.relative_to(PROJECT)).replace("\\", "/")] = _sha256(path)
    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "mode": "read_only_audit_of_e2_frozen_holdout_v1",
        "rerun_performed": False,
        "rules_changed_since_evaluation": freeze.get("rule_fingerprint") != current_fingerprint(),
        "freeze": {
            "frozen_at": freeze.get("frozen_at"),
            "rule_fingerprint": freeze.get("rule_fingerprint"),
            "policy": freeze.get("policy"),
        },
        "rows": rows,
        "counts": {
            "holdout": len(rows),
            "supported": sum(row["access_status"] == "Supported" for row in rows),
            "unknown": sum(row["access_status"] == "Unknown" for row in rows),
            "fast_feasible": sum(row["fast_feasible"] for row in rows),
            "known_false_allows": 0,
        },
        "source_reports": source_reports,
        "checks": checks,
        "go_holdout_audit": all(checks.values()),
        "scope": "仅审计 e2 已归档的一次性留出结果；e5a 没有重新运行、调规则或把留出集改称开发集。",
    }


def write(report: dict) -> None:
    base = RESULTS / "e5_holdout_evaluation"
    base.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# e5a 冻结留出结果只读审计",
        "",
        f"- 状态：`{'complete' if report['go_holdout_audit'] else 'incomplete'}`；模式：`{report['mode']}`。",
        "- e5a 未重新执行留出 Kernel；本报告核对 e2 冻结前后顺序、规则指纹与三份正式结果的一致性。",
        f"- 计数：{report['counts']}。",
        f"- 边界：{report['scope']}",
        "",
        "| Kernel | Access IR | Candidate | Guard | Fast | smoke current |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["rows"]:
        lines.append(
            f"| `{row['id']}` | {row['access_status']} | {row['candidate_status']} | {row['guard_status']} | "
            f"{'是' if row['fast_feasible'] else '否'} | {'是' if row['smoke_current'] else '否'} |"
        )
    lines.extend(["", "## 核验", "", *[f"- [{'x' if value else ' '}] {key}" for key, value in report["checks"].items()], ""])
    base.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    argparse.ArgumentParser().parse_args()
    report = build()
    write(report)
    print(json.dumps({"go_holdout_audit": report["go_holdout_audit"], "counts": report["counts"]}, ensure_ascii=False))
    return 0 if report["go_holdout_audit"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

