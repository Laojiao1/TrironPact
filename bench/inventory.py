"""从登记清单和当前源码生成可审计的 AST 选型摸底报告。"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import textwrap
from collections import Counter
from datetime import datetime
from pathlib import Path

from pact.access_ir import parse_access_ir
from bench.specs import AUDIT_RECORDS


ROOT = Path(__file__).resolve().parent.parent
CATALOG = Path(__file__).with_name("catalog.json")


def function_source(path: Path, name: str) -> tuple[str, int]:
    content = path.read_text(encoding="utf-8")
    tree = ast.parse(content, filename=str(path))
    matches = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(matches) != 1:
        raise ValueError(f"函数缺失或重复：{path}:{name}")
    node = matches[0]
    first = min((item.lineno for item in node.decorator_list), default=node.lineno)
    source = "\n".join(content.splitlines()[first - 1:node.end_lineno])
    return textwrap.dedent(source), first


def build(catalog: Path = CATALOG, root: Path = ROOT) -> dict:
    data = json.loads(catalog.read_text(encoding="utf-8"))
    if data.get("version") != 1 or not isinstance(data.get("entries"), list):
        raise ValueError("基准清单版本或条目无效")
    ids: set[str] = set()
    bodies: set[str] = set()
    rows = []
    if set(AUDIT_RECORDS) != {item["id"] for item in data["entries"]}:
        raise ValueError("基准清单与独立审计规格 ID 不一致")
    for item in data["entries"]:
        if item["id"] in ids:
            raise ValueError(f"重复 Kernel ID：{item['id']}")
        ids.add(item["id"])
        path = (root / item["file"]).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError(f"源码不在项目内：{item['id']}")
        source, line = function_source(path, item["function"])
        digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
        if digest in bodies:
            raise ValueError(f"重复函数体：{item['id']}")
        bodies.add(digest)
        parsed = parse_access_ir(source, item["pointers"], {key: 4 for key in item["pointers"]})
        expected = item.get("expected_parse")
        if expected not in {"Supported", "Unknown", "Unsupported"}:
            raise ValueError(f"缺少预期解析状态：{item['id']}")
        if parsed.status != expected:
            raise ValueError(f"解析状态漂移：{item['id']} expected={expected} actual={parsed.status}")
        audit = AUDIT_RECORDS[item["id"]].to_dict()
        if item["origin"] != "project" and (not item["source_url"] or item["source_revision"] == "unpinned" or item["license_review"] != "approved"):
            raise ValueError(f"外部来源不可追溯：{item['id']}")
        rows.append({**item, "audit": audit, "line": line, "sha256": digest, "parse_status": parsed.status,
                     "parse_reason": parsed.reason, "accesses": len(parsed.accesses), "hints": len(parsed.hints)})
    counts = Counter(row["parse_status"] for row in rows)
    sources = sorted({row["origin"] for row in rows if row["origin"] != "project" and row["source_revision"] != "unpinned" and row["license_review"] == "approved"})
    positive = [row for row in rows if row.get("benchmark_role", "positive") == "positive" and row["parse_status"] == "Supported"]
    positive_families = Counter(row["semantic_family"] for row in positive)
    return {"generated_at": datetime.now().astimezone().isoformat(), "catalog_version": 1,
            "entries": rows, "selection_leads": data.get("selection_leads", []),
            "counts": {"registered": len(rows), "syntax_supported": counts["Supported"],
                       "syntax_unknown": counts["Unknown"], "syntax_unsupported": counts["Unsupported"],
                       "candidate_complete": sum(row["audit"]["candidate_status"] in {"complete", "shadow_complete"} for row in rows),
                       "guard_available": sum(row["audit"]["guard_status"] == "available" for row in rows),
                       "fast_feasible": sum(row["audit"]["fast_status"] == "feasible" for row in rows),
                       "pinned_external_sources": len(sources), "semantic_families": sorted(positive_families),
                       "supported_positive_by_family": dict(sorted(positive_families.items())),
                       "access_families": sorted({row["access_family"] for row in rows})},
            "scope": "仅核对清单、源码指纹和 AST 可解析性；positive/challenge 分列。未执行语义/绑定/Guard/GPU 核验，外部未固定版本不计入合格来源。"}


def render(data: dict) -> str:
    count = data["counts"]
    lines = ["# 第六阶段基准选型摸底", "", f"生成时间：{data['generated_at']}", "",
             f"登记 {count['registered']} 个不同函数体；AST 可解析 {count['syntax_supported']}，Unknown {count['syntax_unknown']}，Unsupported {count['syntax_unsupported']}；已固定且审计许可的外部来源 {count['pinned_external_sources']} 个。",
             f"五层分母：登记 {count['registered']}，AST 可解析 {count['syntax_supported']}，候选完整 {count['candidate_complete']}，Guard 可用 {count['guard_available']}，Fast 实际可行 {count['fast_feasible']}。",
             f"正向可解析语义类计数：{count['supported_positive_by_family']}。", "", "| ID | 角色 | 语义类 | 来源 | AST | 候选 | Guard | Fast |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in data["entries"]:
        audit = row["audit"]
        lines.append(f"| {row['id']} | {row.get('benchmark_role', 'positive')} | {row['semantic_family']} | {row['origin']} | {row['parse_status']} | {audit['candidate_status']} | {audit['guard_status']} | {audit['fast_status']} |")
    lines.extend(["", data["scope"], "", "## 待核查来源（不计入基准）", ""])
    lines += [f"- {item['source']}：{item['url']}；{item['reason']}" for item in data["selection_leads"]]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/phase6_bench_inventory.md")
    args = parser.parse_args()
    data = build()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(data), encoding="utf-8")
    args.output.with_suffix(".json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"registered={data['counts']['registered']} syntax_supported={data['counts']['syntax_supported']} pinned_sources={data['counts']['pinned_external_sources']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
