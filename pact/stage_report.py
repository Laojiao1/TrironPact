"""汇总 Access IR、契约 DSL 和拒绝样例的可复核报告。"""

from datetime import datetime

from pact.access_ir import parse_access_ir
from pact.analysis import extract
from pact.semantics import SEMANTICS
from scenarios.external_add import add_kernel
from scenarios.kernels import KERNELS


def build_stage_report() -> dict:
    cases = {}
    for name in ("A", "A2", "B", "D"):
        meaning = SEMANTICS[name]
        kernel = add_kernel if name == "D" else KERNELS[name]
        pointers = {item.pointer: item.tensor for item in meaning.inputs}
        pointers[meaning.output_pointer] = "OUT"
        ir = parse_access_ir(kernel, pointers, {pointer: 4 for pointer in pointers})
        analysis = extract(name)
        cases[name] = {"ir": ir.to_dict(), "contract_status": analysis.status, "contract_reason": analysis.reason, "clauses": [item.to_dict() for item in analysis.clauses]}

    source = """
def sample(x, out, n, block: tl.constexpr):
    lane = tl.program_id(0) * block + tl.arange(0, block)
    valid = lane < n
    value = tl.load(x + lane, mask=valid, other=0)
    tl.store(out + lane, value + 1, mask=valid)
"""
    pointers = {"x": "X", "out": "OUT"}
    rejected = {
        "缺少 load mask": parse_access_ir(source.replace("mask=valid, other=0", "other=0"), pointers).to_dict(),
        "索引量相乘": parse_access_ir(source.replace("x + lane", "x + lane * lane"), pointers).to_dict(),
        "错误指针绑定": parse_access_ir(source, {"missing": "X", "out": "OUT"}).to_dict(),
    }
    checks = {
        "四例由同一源码解析入口形成 Access IR": all(item["ir"]["status"] == "Supported" for item in cases.values()),
        "四例仍取得 PoC 有限契约": all(item["contract_status"] == "Supported" for item in cases.values()),
        "每条 DSL 条件有来源和适用域": all(clause["origin"]["source_line"] > 0 and clause["domain"] for item in cases.values() for clause in item["clauses"]),
        "拒绝样例没有获得受支持状态": all(item["status"] != "Supported" for item in rejected.values()),
    }
    return {"cases": cases, "rejected": rejected, "checks": checks, "go_ir": all(checks.values())}


def render_stage_report(data: dict) -> str:
    lines = [
        "# TritonPact 契约 DSL 与 Access IR 验收报告",
        "",
        f"> 生成时间：{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}  ",
        f"> 表示与解析阶段检查：**{'通过' if data['go_ir'] else '未通过'}**",
        "",
        "## 检查项",
        "",
        "| 检查项 | 结果 |",
        "| --- | --- |",
    ]
    lines.extend(f"| {name} | {'通过' if passed else '失败'} |" for name, passed in data["checks"].items())
    lines.extend(["", "## 四个 PoC 案例的统一 Access IR", "", "| 案例 | 分析状态 | 访存 | 指针绑定 | 规范化地址 | 规范化 mask | 源码位置 |", "| --- | --- | --- | --- | --- | --- | --- |"])
    for name, item in data["cases"].items():
        ir = item["ir"]
        for point in ir["accesses"]:
            offset = _expr_text(point["offset"])
            mask = _expr_text(point["mask"])
            lines.append(f"| {name} | {ir['status']} | {point['kind']} | {point['pointer_param']} → {point['tensor']} | `{offset}` | `{mask}` | `{point['source_file']}:{point['line']}` |")
    lines.extend(["", "IR 中同时保存原始与展开后的地址、mask、元素宽度及 Tile 参数；完整结构见同名 JSON。`Supported` 只表示访存语法可解释，不授予 Fast 资格。", "", "## 与独立语义绑定的 PoC 条件", "", "| 案例 | 契约状态 | 条件用途 | 逻辑读取与访存来源 |", "| --- | --- | --- | --- |"])
    for name, item in data["cases"].items():
        for clause in item["clauses"]:
            origin = clause["origin"]
            lines.append(f"| {name} | {item['contract_status']} | {clause['purpose']} | `{origin['semantic_index']}` ← `{origin['source_file']}:{origin['source_line']}` |")
    lines.extend(["", "这些是旧 PoC 规则在新 IR 上核对后编码成 DSL 的条件；本阶段没有系统提取 alignment、shape/stride 或 offset/span 新候选。", "", "## 拒绝样例", "", "| 输入 | 状态 | 原因 |", "| --- | --- | --- |"])
    lines.extend(f"| {name} | {item['status']} | {item['reason']} |" for name, item in data["rejected"].items())
    lines.extend(["", "## 结论边界", "", "- 逻辑语义和 launch 参数映射仍由 PoC 的独立声明提供，不从 Kernel 实现推断算子意图。", "- 本报告验证表示、解析与拒绝边界；数值、路径和十次重复的证据仍以 PoC 隔离报告为准。", "- 复杂间接索引、动态控制流、`tl.multiple_of`、库级自动发现及系统候选提取均未在本阶段放行。", ""])
    return "\n".join(line.rstrip() for line in lines)


def _expr_text(data: dict) -> str:
    value = data["value"]
    if value is not None:
        return f"{data['op']}({value})"
    return f"{data['op']}({', '.join(_expr_text(item) for item in data['args'])})"
