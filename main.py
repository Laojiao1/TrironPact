""" CLI 命令行入口（ir 查看中间表示，demo 运行完整验收闭环） """

import argparse
import json
from pathlib import Path

from pact.analysis import extract
from pact.oracle import run_isolated
from pact.refine import refine_singleton_strides
from pact.report import render_markdown
from pact.stage_report import build_stage_report, render_stage_report
from pact.access_ir import parse_access_ir
from pact.semantics import SEMANTICS
from scenarios.external_add import add_kernel
from scenarios.kernels import KERNELS


def demo(repeats: int = 1) -> dict:
    cases = [
        ("A", "contiguous", 7, 11, "dispatch"),
        ("A", "transpose", 7, 11, "raw"),
        ("A", "transpose", 5, 13, "raw"),
        ("A", "transpose", 7, 11, "dispatch"),
        ("A", "padded", 7, 11, "dispatch"),
        ("A2", "padded", 7, 11, "dispatch"),
        ("A2", "padded", 5, 13, "dispatch"),
        ("A2", "transpose", 7, 11, "dispatch"),
        ("B", "contiguous", 7, 127, "dispatch"),
        ("B", "contiguous", 7, 128, "dispatch"),
        ("B", "contiguous", 7, 129, "dispatch"),
        ("B", "offset", 7, 129, "dispatch"),
        ("B", "strided", 7, 129, "dispatch"),
        ("A", "unknown_dtype", 7, 11, "dispatch"),
        ("D", "contiguous", 7, 127, "dispatch"),
        ("D", "contiguous", 7, 129, "dispatch"),
        ("D", "x_strided", 7, 129, "raw"),
        ("D", "x_strided", 7, 129, "dispatch"),
        ("D", "y_strided", 7, 129, "raw"),
        ("D", "y_strided", 7, 129, "dispatch"),
        ("D", "x_offset", 7, 129, "dispatch"),
        ("D", "unknown_dtype", 7, 129, "dispatch"),
    ]
    critical = [("A", "transpose", 7, 11, "raw"), ("A2", "padded", 7, 11, "dispatch"), ("B", "contiguous", 7, 129, "dispatch")]
    def run_case(name: str, layout: str, m: int, n: int, mode: str = "dispatch", *, refined: bool = False) -> dict:
        return run_isolated(name, layout, m=m, n=n, mode=mode, refined=refined).to_dict()

    # 先用原始 Fast 检查单谓词补集，再决定是否生成条件式 Guard。
    # 1. 收集单行单列两个见证用例 (Witnesses)
    row_witness = run_case("A", "single_row", 1, 11, "raw")
    col_witness = run_case("A", "single_col", 7, 1, "raw")

    # 2. 触发契约修订 (Refine)
    revised_a, refinement = refine_singleton_strides(extract("A"), [row_witness, col_witness])
    analysis = {name: (revised_a if name == "A" else extract(name)).to_dict() for name in ("A", "A2", "B", "D")}

    # 3. 运行所有的布局测试用例 (转置、padding、步长、未知类型等)
    runs = [run_case(name, layout, m, n, mode) for name, layout, m, n, mode in cases]
    before = [run_case("A", "single_row", 1, 11), run_case("A", "single_col", 7, 1)]
    after = [run_case("A", "single_row", 1, 11, refined=refinement["changed"]), run_case("A", "single_col", 7, 1, refined=refinement["changed"])]
    runs.extend([row_witness, col_witness, *before, *after])
    repeated = [run_case(name, layout, m, n, mode) for _ in range(repeats - 1) for name, layout, m, n, mode in critical]

    # 验收项同时检查结果与路径，避免回退碰巧算对却被当作 Fast 成功。
    def find(name: str, layout: str, mode: str, n: int | None = None) -> dict:
        return next((run for run in runs if run["detail"].get("case") == name and run["detail"].get("layout") == layout and run["detail"].get("mode") == mode and (n is None or run["detail"].get("shape", [None])[-1] == n)), {"category": "unknown", "detail": {}})

    def has_auditable_sources(item: dict) -> bool:
        meaning = item["semantics"]
        if not (meaning["index_domain"] and meaning["logical_write"] and meaning["reference_symbol"] and meaning["launch_bindings"]):
            return False
        for predicate in item["predicates"]:
            matching_inputs = [entry for entry in meaning["inputs"] if entry["tensor"] == predicate["tensor"]]
            if len(matching_inputs) != 1:
                return False
            source = matching_inputs[0]
            if predicate["semantic_input"] != source["pointer"] or predicate["semantic_index"] != source["logical_read"]:
                return False
            if not any(access["kind"] == "load" and access["tensor"] == source["pointer"] and access["line"] == predicate["source_access_line"] and access["source_file"] == predicate["source_access_file"] for access in item["accesses"]):
                return False
        return True

    checks = {
        "四个变体进入 Access IR": all(item["status"] == "Supported" for item in analysis.values()),
        "语义规格与逐谓词来源可追溯": all(has_auditable_sources(item) for item in analysis.values()),
        "A 两组转置尺寸均出现数值错误": all(find("A", "transpose", "raw", n)["category"] == "numeric_mismatch" for n in (11, 13)),
        "A Guard 拦截并正确回退": find("A", "transpose", "dispatch")["category"] == "correct" and find("A", "transpose", "dispatch")["detail"].get("path") == "PyTorch Fallback",
        "A2 非连续输入正确直通且连续化会复制": all(find("A2", "padded", "dispatch", n)["category"] == "correct" and find("A2", "padded", "dispatch", n)["detail"].get("path") == "Fast" and find("A2", "padded", "dispatch", n)["detail"].get("contiguous_copied") is True for n in (11, 13)),
        "B 非整除边界正确直通": all(find("B", "contiguous", "dispatch", n)["category"] == "correct" and find("B", "contiguous", "dispatch", n)["detail"].get("path") == "Fast" for n in (127, 129)),
        "未知类型安全回退": find("A", "unknown_dtype", "dispatch")["category"] == "correct" and find("A", "unknown_dtype", "dispatch")["detail"].get("path") == "PyTorch Fallback",
        "官方教程双输入契约阻止错读": all(find("D", "contiguous", "dispatch", n)["category"] == "correct" and find("D", "contiguous", "dispatch", n)["detail"].get("path") == "Fast" for n in (127, 129)) and all(find("D", layout, "raw")["category"] == "numeric_mismatch" and find("D", layout, "dispatch")["category"] == "correct" and find("D", layout, "dispatch")["detail"].get("path") == "PyTorch Fallback" for layout in ("x_strided", "y_strided")),
        "关键输入重复稳定": all(
            (run["category"] == "numeric_mismatch" if run["detail"].get("mode") == "raw" else run["category"] == "correct" and run["detail"].get("path") == "Fast")
            for run in repeated
        ),
        "经验观察与静态放行分开记录": all(run.get("evidence", {}).get("observation_level") == "Empirically-Validated" for run in runs if run["category"] in ("correct", "numeric_mismatch")) and find("B", "contiguous", "dispatch", 129).get("evidence", {}).get("fast_eligible") is True and find("A", "unknown_dtype", "dispatch").get("evidence", {}).get("fast_eligible") is False and row_witness.get("evidence", {}).get("fast_eligible") is False,
    }
    refinement_verified = refinement["changed"] and all(run["category"] == "correct" and run["detail"].get("path") == "PyTorch Fallback" for run in before) and all(run["category"] == "correct" and run["detail"].get("path") == "Fast" for run in after)
    checks["单谓词补集触发候选修订并放行"] = refinement_verified
    return {
        "analysis": analysis,
        "runs": runs,
        "repeat_runs": repeated,
        "checks": checks,
        "go_core": all(checks.values()),
        "refinement_verified": refinement_verified,
        "refinement": refinement,
        "refinement_note": "单行和单列的补集输入触发过强候选修订；条件式还经过有限索引范围推导。",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="TritonPact 访存契约分析与隔离验收")
    parser.add_argument("command", choices=("ir", "access-ir", "stage-ir", "demo"))
    parser.add_argument("--kernel", choices=("A", "A2", "B", "D"), default="A")
    parser.add_argument("--output", help="可选：写入 .md 报告或 .json 原始数据")
    parser.add_argument("--repeats", type=int, default=1, help="关键用例总重复次数，范围 1～20")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 20:
        parser.error("--repeats 必须在 1～20 之间")
    if args.command == "ir":
        payload = extract(args.kernel).to_dict()
    elif args.command == "access-ir":
        meaning = SEMANTICS[args.kernel]
        pointers = {item.pointer: item.tensor for item in meaning.inputs}
        pointers[meaning.output_pointer] = "OUT"
        kernel = add_kernel if args.kernel == "D" else KERNELS[args.kernel]
        payload = parse_access_ir(kernel, pointers).to_dict()
    elif args.command == "stage-ir":
        payload = build_stage_report()
    else:
        payload = demo(args.repeats)
    rendered_json = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        if output.suffix == ".md" and args.command in ("demo", "stage-ir"):
            output.write_text((render_stage_report if args.command == "stage-ir" else render_markdown)(payload), encoding="utf-8")
            # JSON 留作机器复核；日常阅读只需打开 Markdown。
            output.with_suffix(".json").write_text(rendered_json, encoding="utf-8")
        elif output.suffix == ".json":
            output.write_text(rendered_json, encoding="utf-8")
        else:
            parser.error("demo 报告请使用 .md；原始数据请使用 .json")
    else:
        print(render_markdown(payload) if args.command == "demo" else (render_stage_report(payload) if args.command == "stage-ir" else rendered_json))
    if args.command == "demo":
        return 0 if payload["go_core"] else 1
    if args.command == "stage-ir":
        return 0 if payload["go_ir"] else 1
    return 0 if payload["status"] == "Supported" else 1


if __name__ == "__main__":
    raise SystemExit(main())
