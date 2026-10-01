"""为 e2 开发集生成逐访问点候选与有界 span 报告。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime

from bench.e1.catalog import load
from bench.e2.coverage import DTYPE_BYTES, RESULTS, _constexpr_values, _element_bytes, _kernel
from pact.e2_access import parse_extended_access_ir
from pact.e2_contracts import FrozenTensor, contiguous_stride, extract_bounded_contracts


def _tensors(entry: dict) -> dict[str, FrozenTensor]:
    return {
        item["name"]: FrozenTensor(
            item["name"],
            tuple(item["shape"]),
            contiguous_stride(tuple(item["shape"])),
            DTYPE_BYTES[item["dtype"]],
            item["dtype"],
        )
        for item in entry["smoke"]["tensors"]
    }


def analyze(evaluate_holdout: bool = False) -> dict:
    allowed = {"development"} | ({"holdout"} if evaluate_holdout else set())
    rows: list[dict] = []
    for entry in load()["entries"]:
        if entry["split"] not in allowed or not entry["smoke"]:
            continue
        kernel = _kernel(entry)
        ir = parse_extended_access_ir(kernel, entry["pointers"], _element_bytes(entry), _constexpr_values(entry, kernel))
        bindings = tuple((item["parameter"], item["expression"]) for item in entry["smoke"]["bindings"])
        contract = extract_bounded_contracts(
            ir,
            _tensors(entry),
            bindings,
            tuple(entry["smoke"]["grid"]),
            entry["reference_anchor"],
        )
        complete = (
            contract.status == "Supported"
            and len(contract.spans) == len(ir.accesses)
            and len(contract.candidates) == 2 * len(ir.accesses)
            and all(candidate.origin and candidate.binding and candidate.domain for candidate in contract.candidates)
        )
        rows.append(
            {
                "id": entry["id"],
                "source": entry["source"],
                "family": entry["family"],
                "split": entry["split"],
                "ir_status": ir.status,
                "status": "complete" if complete else contract.status,
                "reason": contract.reason,
                "access_count": len(ir.accesses),
                "spans": [list(span) for span in contract.spans],
                "candidates": [candidate.to_dict() for candidate in contract.candidates],
                "numeric_semantics": "not_proven; independent frozen smoke is empirical evidence only",
            }
        )
    complete_rows = [row for row in rows if row["status"] == "complete"]
    counts = {
        "total": len(rows),
        "candidate_complete": len(complete_rows),
        "candidate_incomplete": len(rows) - len(complete_rows),
        "candidates": sum(len(row["candidates"]) for row in rows),
    }
    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "mode": "development_and_frozen_holdout" if evaluate_holdout else "development_only",
        "holdout_used_for_rule_development": False,
        "counts": counts,
        "complete_by_family": dict(Counter(row["family"] for row in complete_rows)),
        "complete_by_source": dict(Counter(row["source"] for row in complete_rows)),
        "rows": rows,
        "scope": "候选只覆盖冻结 wrapper/grid 的物理映射与访存 span；reduction/普通算术数值语义未被静态证明。",
    }


def write_report(data: dict) -> None:
    path = RESULTS / "e2_candidate_extraction"
    path.with_suffix(".json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# e2 候选提取",
        "",
        f"- 模式：`{data['mode']}`。",
        f"- 计数：{data['counts']}。",
        f"- 类别：{data['complete_by_family']}。",
        f"- 边界：{data['scope']}",
        "",
        "| Kernel | 来源 | 类别 | 状态 | 访问点 | span |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in data["rows"]:
        lines.append(f"| {row['id']} | {row['source']} | {row['family']} | {row['status']} | {row['access_count']} | {row['spans']} |")
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate-holdout", action="store_true")
    args = parser.parse_args()
    data = analyze(args.evaluate_holdout)
    write_report(data)
    print(json.dumps(data["counts"], ensure_ascii=False))
    return 0 if data["counts"]["candidate_incomplete"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
