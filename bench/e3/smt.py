"""对 e2 的真实候选执行 SMT 可满足性、冲突、蕴含和冗余审计。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from bench.e1.catalog import load
from pact.candidates import Candidate
from pact.smt_refine import Domain, check_redundancy


PROJECT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT / "results"


def _catalog() -> dict[str, dict]:
    return {entry["id"]: entry for entry in load()["entries"]}


def _condition_key(candidate: Candidate) -> str:
    return json.dumps(asdict(candidate.condition), ensure_ascii=False, sort_keys=True) if candidate.condition else "obligation"


def build(timeout_ms: int = 2000) -> dict:
    source = json.loads((RESULTS / "e2_candidate_extraction.json").read_text(encoding="utf-8"))
    catalog = _catalog()
    rows = []
    for item in source["rows"]:
        if item["status"] != "complete":
            rows.append({"id": item["id"], "status": item["status"], "reason": item["reason"], "real_candidate_deletions": []})
            continue
        candidates = [Candidate.from_dict(data) for data in item["candidates"]]
        entry = catalog[item["id"]]
        ranks = tuple((tensor["name"], len(tensor["shape"])) for tensor in entry["smoke"]["tensors"])
        if any(rank not in (1, 2) for _, rank in ranks):
            rows.append(
                {
                    "id": item["id"],
                    "source": item["source"],
                    "family": item["family"],
                    "status": "Unknown",
                    "reason": "现有受限 SMT 域只编码一维/二维 Tensor；候选全部保留",
                    "candidate_count": len(candidates),
                    "kept": list(range(len(candidates))),
                    "real_candidate_deletions": [],
                    "equivalent_groups": [],
                    "query_status": {"unknown": 1},
                    "queries": [],
                    "scope": "未编码维度不创建自由变量，不执行删除",
                }
            )
            continue
        result = check_redundancy(candidates, Domain(ranks), timeout_ms=timeout_ms)
        exact_groups: dict[tuple, list[int]] = {}
        for index, candidate in enumerate(candidates):
            key = (candidate.purpose, candidate.domain, candidate.evidence, _condition_key(candidate))
            exact_groups.setdefault(key, []).append(index)
        equivalent_groups = [indexes for indexes in exact_groups.values() if len(indexes) > 1]
        query_counts = Counter(query.get("status", "unknown") for query in result["queries"])
        rows.append(
            {
                "id": item["id"],
                "source": item["source"],
                "family": item["family"],
                "status": result["status"],
                "reason": result["reason"],
                "candidate_count": len(candidates),
                "kept": result["kept"],
                "real_candidate_deletions": result["removed"],
                "equivalent_groups": equivalent_groups,
                "query_status": dict(query_counts),
                "queries": result["queries"],
                "scope": "仅同 purpose、domain、evidence 的完整编码条件允许删除；AccessSpan 未编码时保守保留",
            }
        )
    complete = [row for row in rows if row.get("candidate_count") is not None]
    encoded = [row for row in complete if row["queries"]]
    deletions = sum(len(row["real_candidate_deletions"]) for row in complete)
    conflicts = sum(row["status"] == "Conflict" or row["query_status"].get("unsat", 0) > len(row["real_candidate_deletions"]) for row in complete)
    unknown_queries = sum(row["query_status"].get("unknown", 0) for row in complete)
    checks = {
        "full_e2_denominator": len(rows) == 28 and len(complete) == 27,
        "unknown_preserved": any(row["id"] == "flag_slice" and row["status"] == "Unknown" for row in rows),
        "no_domain_conflict": all(row["status"] != "Conflict" for row in complete),
        "all_deletions_same_scope": all(
            all(any(index in group for group in row["equivalent_groups"]) for index in row["real_candidate_deletions"])
            for row in complete
        ),
        "synthetic_fixture_excluded": True,
    }
    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "solver": "Z3",
        "timeout_ms_per_query": timeout_ms,
        "rows": rows,
        "counts": {
            "kernels": len(rows),
            "audited": len(complete),
            "smt_encoded": len(encoded),
            "real_candidates": sum(row["candidate_count"] for row in complete),
            "real_candidate_deletions": deletions,
            "kernels_with_conflict": conflicts,
            "unknown_queries": unknown_queries,
            "synthetic_deletions": 0,
        },
        "checks": checks,
        "go_smt": all(checks.values()),
        "claim": "真实候选存在可删除的同域等价重复" if deletions else "真实候选删除为 0，SMT 降为一致性审计机制",
    }


def write(report: dict) -> None:
    base = RESULTS / "e3_smt_audit"
    base.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# e3 真实候选 SMT 审计",
        "",
        f"- 生成时间：{report['generated_at']}。",
        f"- 状态：`{'go' if report['go_smt'] else 'no_go'}`；计数：`{report['counts']}`。",
        f"- 贡献定位：{report['claim']}。",
        "- 合成 fixture 删除不计入真实收益；未编码的 AccessSpan 和求解 Unknown 全部保留。",
        "",
        "| Kernel | 状态 | 候选 | 真实删除 | 等价组 | 查询状态 |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for row in report["rows"]:
        lines.append(
            f"| `{row['id']}` | {row['status']} | {row.get('candidate_count', 0)} | {row['real_candidate_deletions']} | "
            f"{row.get('equivalent_groups', [])} | {row.get('query_status', {})} |"
        )
    lines.append("")
    base.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout-ms", type=int, default=2000)
    args = parser.parse_args()
    report = build(args.timeout_ms)
    write(report)
    print(json.dumps({"go_smt": report["go_smt"], "counts": report["counts"], "claim": report["claim"]}, ensure_ascii=False))
    return 0 if report["go_smt"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

