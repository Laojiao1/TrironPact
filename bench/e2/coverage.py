"""在冻结语料上生成 e2 Access IR 覆盖报告。"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from bench.e1.acquire import ROOT as E1_ROOT
from pact.access_ir import _source
from pact.e2_access import parse_extended_access_ir

PROJECT = E1_ROOT.parent.parent
RESULTS = PROJECT / "results"
CATALOG = E1_ROOT / "catalog.json"
DTYPE_BYTES = {"bool": 1, "float16": 2, "bfloat16": 2, "float32": 4, "int32": 4, "int64": 8}


def _digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.relative_to(PROJECT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _kernel(entry: dict):
    module_name = "bench.e1." + entry["kernel_module"].removesuffix(".py").replace("/", ".")
    module = importlib.import_module(module_name)
    return getattr(module, entry["function"].split(".")[-1])


def _constexpr_names(kernel) -> set[str]:
    source, _, _ = _source(kernel)
    function = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef))
    result: set[str] = set()
    for arg in function.args.args:
        annotation = arg.annotation
        if (
            isinstance(annotation, ast.Attribute)
            and isinstance(annotation.value, ast.Name)
            and annotation.value.id == "tl"
            and annotation.attr == "constexpr"
        ) or (isinstance(annotation, ast.Constant) and annotation.value == "tl.constexpr"):
            result.add(arg.arg)
    return result


def _constexpr_values(entry: dict, kernel) -> dict[str, int | bool]:
    names = _constexpr_names(kernel)
    result: dict[str, int | bool] = {}
    for binding in entry["smoke"]["bindings"]:
        if binding["parameter"] not in names:
            continue
        try:
            value = ast.literal_eval(binding["expression"])
        except (SyntaxError, ValueError):
            continue
        if type(value) in (int, bool):
            result[binding["parameter"]] = value
    function = getattr(kernel, "fn", kernel)
    for name, value in function.__globals__.items():
        if name in names or not name.startswith("_"):
            continue
        if type(value) in (int, bool):
            result[name] = value
            continue
        if value.__class__.__name__ == "constexpr":
            try:
                result[name] = int(value)
            except (TypeError, ValueError):
                pass
    return result


def _element_bytes(entry: dict) -> dict[str, int]:
    tensors = {item["name"]: item for item in entry["smoke"]["tensors"]}
    result: dict[str, int] = {}
    for pointer, tensor in entry["pointers"].items():
        dtype = tensors[tensor]["dtype"]
        if dtype not in DTYPE_BYTES:
            raise ValueError(f"{entry['id']} 的 dtype 未登记元素宽度：{dtype}")
        result[pointer] = DTYPE_BYTES[dtype]
    return result


def analyze(evaluate_holdout: bool = False) -> dict:
    """分析开发集；只有显式授权时才首次纳入冻结留出集。"""

    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    allowed = {"development"} | ({"holdout"} if evaluate_holdout else set())
    entries = [entry for entry in catalog["entries"] if entry["split"] in allowed and entry["smoke"]]
    rows: list[dict] = []
    source_paths = [PROJECT / "pact" / "e2_access.py", Path(__file__), CATALOG]
    for entry in entries:
        kernel = _kernel(entry)
        source_paths.append(E1_ROOT / entry["kernel_module"])
        result = parse_extended_access_ir(
            kernel,
            entry["pointers"],
            _element_bytes(entry),
            _constexpr_values(entry, kernel),
        )
        rows.append(
            {
                "id": entry["id"],
                "source": entry["source"],
                "family": entry["family"],
                "split": entry["split"],
                "status": result.status,
                "reason": result.reason,
                "access_count": len(result.accesses),
                "load_count": sum(point.kind == "load" for point in result.accesses),
                "store_count": sum(point.kind == "store" for point in result.accesses),
                "accesses": [point.to_dict() for point in result.accesses],
                "scope": "访问地址与 mask；不证明普通算术或 reduction 数值语义",
            }
        )
    status = Counter(row["status"] for row in rows)
    by_family = {
        family: Counter(row["status"] for row in rows if row["family"] == family)
        for family in sorted({row["family"] for row in rows})
    }
    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "mode": "development_and_frozen_holdout" if evaluate_holdout else "development_only",
        "holdout_used_for_rule_development": False,
        "fingerprint": _digest(source_paths),
        "counts": {"total": len(rows), **status},
        "by_family": {name: dict(counts) for name, counts in by_family.items()},
        "rows": rows,
        "scope": "Supported 仅表示绑定后的全部访存点进入统一 IR；候选、Guard 和 Fast 资格另行核验。",
    }


def write_report(data: dict) -> None:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / "e2_access_ir_coverage"
    path.with_suffix(".json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# e2 Access IR 覆盖",
        "",
        f"- 模式：`{data['mode']}`。",
        f"- 指纹：`{data['fingerprint']}`。",
        f"- 覆盖：{data['counts']}。",
        f"- 边界：{data['scope']}",
        "",
        "| Kernel | 来源 | 类别 | 划分 | 状态 | load/store | 原因 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in data["rows"]:
        lines.append(
            f"| {row['id']} | {row['source']} | {row['family']} | {row['split']} | {row['status']} | "
            f"{row['load_count']}/{row['store_count']} | {row['reason']} |"
        )
    path.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluate-holdout", action="store_true")
    args = parser.parse_args()
    data = analyze(args.evaluate_holdout)
    write_report(data)
    print(json.dumps(data["counts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
