"""生成主张到代码、测试、报告、原始数据和限制的追踪表。"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from bench.e5.specs import CLAIMS


PROJECT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT / "results"


def build() -> dict:
    rows = []
    for claim in CLAIMS:
        row = claim.to_dict()
        referenced = (*claim.code, *claim.tests, *claim.reports, *claim.raw_data)
        missing = [path for path in referenced if not (PROJECT / path).is_file()]
        row["missing"] = missing
        row["complete"] = not missing and bool(claim.limitations)
        rows.append(row)
    checks = {
        "all_claims_have_code": all(row["code"] for row in rows),
        "all_claims_have_tests": all(row["tests"] for row in rows),
        "all_claims_have_reports": all(row["reports"] for row in rows),
        "all_claims_have_raw_data": all(row["raw_data"] for row in rows),
        "all_claims_state_limitations": all(row["limitations"] for row in rows),
        "all_references_exist": all(row["complete"] for row in rows),
    }
    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "schema_version": 1,
        "rows": rows,
        "counts": {"claims": len(rows), "complete": sum(row["complete"] for row in rows)},
        "checks": checks,
        "go_traceability": all(checks.values()),
    }


def write(report: dict) -> None:
    base = RESULTS / "e5_traceability"
    base.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# e5a 主张与证据追踪",
        "",
        f"- 状态：`{'complete' if report['go_traceability'] else 'incomplete'}`；{report['counts']['complete']}/{report['counts']['claims']} 项引用完整。",
        "- 每项均显式列出证据级别与限制；路径存在只证明资产可定位，不替代其内部结论。",
        "",
        "| ID | 主张 | 证据级别 | 代码 | 测试 | 报告/原始数据 | 限制 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in report["rows"]:
        lines.append(
            f"| `{row['id']}` | {row['claim']} | `{row['evidence_level']}` | {'<br>'.join(row['code'])} | "
            f"{'<br>'.join(row['tests'])} | {'<br>'.join((*row['reports'], *row['raw_data']))} | {'<br>'.join(row['limitations'])} |"
        )
    lines.extend(["", "## 机器核验", "", *[f"- [{'x' if value else ' '}] {key}" for key, value in report["checks"].items()], ""])
    base.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    argparse.ArgumentParser().parse_args()
    report = build()
    write(report)
    print(json.dumps({"go_traceability": report["go_traceability"], "counts": report["counts"]}, ensure_ascii=False))
    return 0 if report["go_traceability"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

