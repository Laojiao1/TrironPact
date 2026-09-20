"""生成详细的 Markdown 验收报告与逐例指标表格"""

from collections import Counter
from datetime import datetime


def _cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _table(headers: tuple[str, ...], rows: list[tuple[object, ...]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(_cell(value) for value in row) + " |" for row in rows)
    return lines


def _outcome(run: dict) -> str:
    category = run["category"]
    detail = run["detail"]
    if category == "correct":
        return "正确"
    if category == "numeric_mismatch" and detail.get("mode") == "raw" and detail.get("case") in ("A", "D"):
        count = detail.get("mismatch_count")
        return f"预期错读（{count} 处）" if count is not None else "预期错读"
    return category


def _predicate(item: dict) -> str:
    condition = f"{item.get('tensor', 'X')}.stride({item['dimension']}) == {item['expected']}"
    axis = item.get("unless_size_one")
    return f"size({axis}) == 1 或 {condition}" if axis is not None else condition


def render_markdown(data: dict) -> str:
    """先给结论，再给逐例证据，最后说明证明范围。"""
    lines = [
        "# TritonPact 受限契约与分派回归报告",
        "",
        f"> 生成时间：{datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}  ",
        f"> 受限案例验收：**{'通过' if data['go_core'] else '未通过'}**  ",
        f"> A 模板内的动态候选修订：**{'已验证' if data['refinement_verified'] else '尚未验证'}**",
        "",
        "## 验收项",
        "",
    ]
    lines.extend(_table(("检查项", "结果"), [(name, "通过" if passed else "失败") for name, passed in data["checks"].items()]))
    lines.extend(["", "## 逐例结果", ""])
    rows = []
    for run in data["runs"]:
        detail = run["detail"]
        mismatch = detail.get("first_mismatch")
        note = ""
        if mismatch is not None:
            note = f"首错 {mismatch}：预期 {detail.get('expected_value')}，实际 {detail.get('actual_value')}"
        elif detail.get("case") == "A2" and detail.get("layout") == "padded":
            note = f"输入非连续；contiguous() {'会复制' if detail.get('contiguous_copied') else '未复制'}"
        elif run["category"] not in ("correct", "numeric_mismatch"):
            note = detail.get("message", run["stderr"][:100])
        mode = "原始 Fast" if detail.get("mode") == "raw" else ("修订后 Guard" if detail.get("refined_contract") else "初始 Guard")
        evidence = run.get("evidence", {})
        guard_basis = evidence.get("guard_basis", "Unknown")
        if guard_basis == "Statically-Proven":
            guard_basis += "（放行）" if evidence.get("fast_eligible") else "（拒绝）"
        rows.append((detail.get("case", "?"), "×".join(map(str, detail.get("shape", ()))), detail.get("layout", "?"), mode, detail.get("path", "未执行"), _outcome(run), guard_basis, evidence.get("observation_level", "Unknown"), note))
    lines.extend(_table(("案例", "尺寸", "布局", "检查方式", "路径", "结果", "Guard 依据", "运行证据", "说明"), rows))
    lines.extend(["", "`Statically-Proven` 是 Guard 在声明语义和支持域内的静态判断；仅当它放行时才有 Fast 资格。`Empirically-Validated` 只表示这一行的执行结果已观察到，不会单独授予 Fast 资格。", ""])

    lines.extend(["", "## 关键案例重复", "", "表中次数包含逐例结果中的首次运行。", ""])
    counts = Counter((run["detail"].get("case", "?"), run["detail"].get("layout", "?"), run["detail"].get("mode", "?"), run["category"], run["detail"].get("path", "未执行")) for run in data["repeat_runs"])
    repeat_rows = [(name, layout, mode, _outcome({"category": category, "detail": {"case": name, "mode": mode}}), path, count + 1) for (name, layout, mode, category, path), count in sorted(counts.items())]
    if repeat_rows:
        lines.extend(_table(("案例", "布局", "模式", "结果", "路径", "总次数"), repeat_rows))
    else:
        lines.append("快速模式：关键案例各运行 1 次。")

    lines.extend(["", "## 提取出的访问与布局条件", ""])
    ir_rows = []
    for name, analysis in data["analysis"].items():
        for load in (access for access in analysis["accesses"] if access["kind"] == "load"):
            tensor = "Y" if load["tensor"] == "y_ptr" else "X"
            predicates = "；".join(_predicate(item) for item in analysis["predicates"] if item.get("tensor", "X") == tensor) or "无"
            ir_rows.append((name, tensor, analysis["status"], f"`{load.get('expanded_offset', '?')}`", f"`{load.get('mask', '?')}`", f"`{predicates}`"))
    lines.extend(_table(("案例", "输入", "分析状态", "输入地址偏移", "读取 mask", "候选条件"), ir_rows))
    lines.extend(["", "## 外部语义输入与谓词来源", "", "算子语义、参数映射和参考实现是分析的外部输入；它们不由 Kernel AST 独自推得，也没有预填目标 stride 谓词。", ""])
    semantic_rows = []
    origin_rows = []
    for name, analysis in data["analysis"].items():
        meaning = analysis["semantics"]
        bindings = "；".join(f"{param} ← {value}" for param, value in meaning["launch_bindings"])
        reads = "；".join(f"{item['pointer']} → {item['logical_read']}，shape={tuple(item['shape_params'])}" for item in meaning["inputs"])
        semantic_rows.append((name, meaning["index_domain"], reads, meaning["logical_write"], bindings, meaning["reference_rule"], meaning["reference_symbol"]))
        for item in analysis["predicates"]:
            origin_rows.append((name, _predicate(item), f"{item['semantic_input']} 的 load：`{item['source_access_file']}:{item['source_access_line']}`", item["semantic_index"], item["source"]))
    lines.extend(_table(("案例", "逻辑索引域", "预期读取及形状", "预期写入", "启动参数映射", "参考计算", "参考入口"), semantic_rows))
    lines.extend(["", "每条布局条件对应的访问点和语义假设：", ""])
    lines.extend(_table(("案例", "布局条件", "访存来源", "语义读取", "推导说明"), origin_rows))
    refinement = data.get("refinement", {})
    lines.extend(["", "## 边界反例与候选修订", ""])
    if refinement.get("changed"):
        lines.extend([
            "A 的初始候选在单行、单列输入上各误拒绝一个正确实例。隔离运行确认实例正确后，有限索引分析给出以下修订：",
            "",
        ])
        change_rows = [(_predicate(before), _predicate(after)) for before, after in zip(refinement["before"], refinement["after"])]
        lines.extend(_table(("初始候选", "修订后条件"), change_rows))
        lines.extend(["", "边界证据：", ""])
        lines.extend(_table(("输入尺寸", "输入 stride", "只违反的原谓词"), [("×".join(map(str, item["shape"])), tuple(item["stride"]), f"stride({item['violated_predicate_dimension']})") for item in refinement["witnesses"]]))
        lines.extend(["", f"推导依据：{refinement['static_reason']}", ""])
    else:
        lines.extend([f"尚未修改候选：{refinement.get('reason', '无边界证据')}。", ""])
    lines.extend([
        "",
        "## 结论边界",
        "",
        "- `通过` 表示本报告中的受限语义、地址模板和输入域达到了 PoC 最低验收，不代表通用 Triton Kernel 安全。",
        "- Case A 的原始 Fast 错读是预期的对照结果；Guard 应阻止相同错误布局进入 Fast。",
        "- A 的单行、单列反例触发了真实候选修改；只有这两个实例的成功仍不足以证明整个补集，条件式还依赖报告列出的静态索引推导。",
        "- Case B 的尾部 mask 在静态分析中已被识别，因此 `N % B == 0` 未进入契约；B 没有发生动态修订。",
        "- Case D 取自 [Triton 官方向量加法教程](https://github.com/triton-lang/triton/blob/main/python/tutorials/01-vector-add.py)，验证了第二个输入指针与 load 的提取；它不是生产项目缺陷案例。",
        "- 本报告记录数值正确性与路径；未测量端到端性能，也未对可选的对齐案例作结论。",
        "- 机器数据把 `guard_basis`、`fast_eligible` 与 `observation_level` 分开保存；即使 Fallback 的输出正确，若静态条件不足，Fast 仍不放行。",
        "",
    ])
    return "\n".join(line.rstrip() for line in lines)
