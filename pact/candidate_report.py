"""第三阶段离线候选与旧 Guard 影子对照报告。"""

from __future__ import annotations

import json
from dataclasses import asdict, replace
from datetime import datetime

from pact.access_ir import parse_access_ir
from pact.alignment import extract_alignment
from pact.call_site import audit_call_site
from pact.candidates import CallBinding, Candidate
from pact.contract_dsl import TensorMetadata
from pact.semantics import SEMANTICS
from pact.shape_stride import extract_shape_stride
from pact.span import extract_spans
from pact.runtime import guard
from scenarios.external_add import add_kernel, run_add
from scenarios.holdouts import MATRIX_MEANING, VECTOR_MEANING, run_strided_vector, run_two_stride_matrix, strided_vector, two_stride_matrix
from scenarios.kernels import KERNELS, run_fast


def _sample(shape, stride, offset, nbytes, ptr=0x1000):
    return TensorMetadata(tuple(shape), tuple(stride), offset, ptr, 4, nbytes)


SAMPLES = {
    "A": ({"X": _sample((7, 11), (11, 1), 0, 308), "OUT": _sample((7, 11), (11, 1), 0, 308, 0x2000)}, {"M": 7, "N": 11, "B": 128}),
    "A2": ({"X": _sample((7, 11), (14, 1), 0, 392), "OUT": _sample((7, 11), (11, 1), 0, 308, 0x2000)}, {"M": 7, "N": 11, "S0": 14, "B": 16}),
    "B": ({"X": _sample((129,), (1,), 1, 520, 0x1004), "OUT": _sample((129,), (1,), 0, 516, 0x2000)}, {"N": 129, "B": 128}),
    "D": ({"X": _sample((129,), (1,), 1, 520, 0x1004), "Y": _sample((129,), (1,), 0, 516, 0x3000), "OUT": _sample((129,), (1,), 0, 516, 0x2000)}, {"n_elements": 129, "BLOCK_SIZE": 128}),
    "holdout_vector": ({"X": _sample((129,), (2,), 0, 1032), "OUT": _sample((129,), (1,), 0, 516, 0x2000)}, {"N": 129, "S": 2, "B": 128}),
    "holdout_matrix": ({"X": _sample((7, 11), (25, 2), 0, 700), "OUT": _sample((7, 11), (11, 1), 0, 308, 0x2000)}, {"M": 7, "N": 11, "S0": 25, "S1": 2, "B": 16}),
}


def _case(kernel, wrapper, meaning, sample_name: str) -> dict:
    pointers = {item.pointer: item.tensor for item in meaning.inputs}
    pointers[meaning.output_pointer] = "OUT"
    ir = parse_access_ir(kernel, pointers, {name: 4 for name in pointers})
    bindings = dict(meaning.launch_bindings)
    call = CallBinding(tuple(pointers.items()), tuple((name, value) for name, value in meaning.launch_bindings if name != "grid"), bindings["grid"], meaning.inputs[0].shape_params, 4, "已核对 wrapper 源码")
    site = audit_call_site(wrapper, kernel.fn.__name__, call)
    shape = extract_shape_stride(ir, meaning, call, site)
    span = extract_spans(ir, meaning, call, site)
    alignment = extract_alignment(ir, meaning, call, site)
    tensors, scalars = SAMPLES[sample_name]
    return {
        "ir": ir.to_dict(), "meaning": asdict(meaning), "call": call.to_dict(), "call_site": asdict(site),
        "status": {"shape_stride": shape.status, "span": span.status, "alignment": alignment.status},
        "reason": {"shape_stride": shape.reason, "span": span.reason, "alignment": alignment.reason},
        "candidates": {kind: [item.to_dict() for item in result.candidates] for kind, result in (("shape_stride", shape), ("span", span), ("alignment", alignment))},
        "shadow": {"tensor_metadata": {name: asdict(meta) for name, meta in tensors.items()}, "scalars": scalars, "evaluation": {kind: [item.evaluate(tensors, scalars) for item in result.candidates] for kind, result in (("shape_stride", shape), ("span", span), ("alignment", alignment))}},
    }


def _rejections() -> dict:
    meaning = SEMANTICS["B"]
    kernel = KERNELS["B"]
    pointers = {"X": "X", "Y": "OUT"}
    ir = parse_access_ir(kernel, pointers, {"X": 4, "Y": 4})
    call = CallBinding((("X", "X"), ("Y", "OUT")), (("N", "x.size(0)"), ("B", "128")), "ceil(N/B)", ("N",), 4, "scenarios/kernels.py")
    site = audit_call_site(run_fast, kernel.fn.__name__, call)
    bad = {
        "missing_semantics": (ir, None, call, site),
        "wrong_grid": (ir, meaning, replace(call, grid="N"), site),
        "wrong_pointer": (ir, meaning, replace(call, pointers=(("X", "OUT"), ("Y", "X"))), site),
        "missing_width": (parse_access_ir(kernel, pointers), meaning, call, site),
        "wrong_output": (ir, meaning, replace(call, output_shape=()), site),
    }
    wrong_mask_source = """
def sample(X, Y, N: tl.constexpr, B: tl.constexpr):
    idx = tl.program_id(0) * B + tl.arange(0, B)
    mask = idx < N
    value = tl.load(X + idx, mask=mask, other=0)
    tl.store(Y + idx, value + 1, mask=tl.arange(0, B) < N)
"""
    bad["wrong_mask"] = (parse_access_ir(wrong_mask_source, pointers, {"X": 4, "Y": 4}), meaning, call, site)
    a2_meaning = SEMANTICS["A2"]
    a2_kernel = KERNELS["A2"]
    a2_ir = parse_access_ir(a2_kernel, {"X": "X", "Y": "OUT"}, {"X": 4, "Y": 4})
    a2_call = CallBinding((("X", "X"), ("Y", "OUT")), tuple((key, value) for key, value in a2_meaning.launch_bindings if key != "grid"), "M", ("M", "N"), 4, "scenarios/kernels.py")
    a2_site = audit_call_site(run_fast, a2_kernel.fn.__name__, a2_call)
    bad["wrong_stride"] = (a2_ir, a2_meaning, replace(a2_call, scalars=tuple((key, "x.stride(1)" if key == "S0" else value) for key, value in a2_call.scalars)), a2_site)
    result = {}
    for name, args in bad.items():
        shape = extract_shape_stride(*args)
        span = extract_spans(*args)
        result[name] = {"shape_status": shape.status, "span_status": span.status, "shape_reason": shape.reason, "span_reason": span.reason, "shape_candidates": [item.to_dict() for item in shape.candidates], "span_candidates": [item.to_dict() for item in span.candidates]}
    return result


def _old_path(regression: dict, name: str, layout: str) -> str:
    matches = [run for run in regression.get("runs", ()) if run.get("detail", {}).get("case") == name and run.get("detail", {}).get("layout") == layout and run.get("detail", {}).get("mode") == "dispatch"]
    return matches[0].get("detail", {}).get("path", "Unknown") if matches else "Unknown"


def build_candidate_report(regression: dict) -> dict:
    specs = {
        "A": (KERNELS["A"], run_fast, SEMANTICS["A"]),
        "A2": (KERNELS["A2"], run_fast, SEMANTICS["A2"]),
        "B": (KERNELS["B"], run_fast, SEMANTICS["B"]),
        "D": (add_kernel, run_add, SEMANTICS["D"]),
        "holdout_vector": (strided_vector, run_strided_vector, VECTOR_MEANING),
        "holdout_matrix": (two_stride_matrix, run_two_stride_matrix, MATRIX_MEANING),
    }
    cases = {name: _case(*spec, name) for name, spec in specs.items()}
    hints = {
        "pointer": _case_hint("pointer"),
        "index": _case_hint("index"),
    }
    rejections = _rejections()
    old_paths = {"A2_padded": _old_path(regression, "A2", "padded"), "B_offset": _old_path(regression, "B", "offset"), "D_x_offset": _old_path(regression, "D", "x_offset"), "holdout_vector": guard("holdout_vector", object()).evidence, "holdout_matrix": guard("holdout_matrix", object()).evidence}
    hint_rejection = _hint_rejection()
    records = [candidate for item in (*cases.values(), *hints.values()) for group in item["candidates"].values() for candidate in group]
    records.extend(candidate for item in rejections.values() for group in ("shape_candidates", "span_candidates") for candidate in item[group])
    records.extend(hint_rejection["candidates"])
    totals = {"generated": len(records), "statically_proven": sum(item["evidence"] == "Statically-Proven" for item in records), "unproven": sum(item["evidence"] == "Unproven" for item in records), "unknown": sum(item["evidence"] == "Unknown" for item in records)}
    checks = {
        "六个案例由同一 IR 与候选规则处理": all(item["ir"]["status"] == "Supported" and item["call_site"]["status"] == "Supported" and item["status"]["shape_stride"] == item["status"]["span"] == "Supported" for item in cases.values()),
        "三类候选有逐访问来源和绑定": all(candidate["origin"]["source_line"] > 0 and candidate["binding"] and candidate["rule"] and candidate["reason"] for candidate in records),
        "影子样例的 shape 与 span 条件可求值": all(all(value is True for value in item["shadow"]["evaluation"][kind]) for item in cases.values() for kind in ("shape_stride", "span")),
        "提示区分指针与索引且未知向量不推指针对齐": hints["pointer"]["candidates"]["alignment"][0]["evidence"] == "Unproven" and hints["pointer"]["candidates"]["alignment"][0]["condition"]["left"]["kind"] == "effective_ptr" and hints["index"]["candidates"]["alignment"][0]["condition"]["left"]["kind"] == "scalar" and hint_rejection["status"] == "Unknown" and all(item["condition"] is None for item in hint_rejection["candidates"]),
        "错误绑定与缺失信息拒绝": all(item["shape_status"] == item["span_status"] == "Unknown" for item in rejections.values()),
        "旧隔离回归通过且留出名称无 Fast 路径": regression.get("go_core") is True and old_paths["A2_padded"] == old_paths["B_offset"] == old_paths["D_x_offset"] == "Fast" and all(old_paths[name] == "Unsupported" for name in ("holdout_vector", "holdout_matrix")),
    }
    return {"generated_at": datetime.now().astimezone().isoformat(), "phase": "第三阶段候选提取；离线/影子，不接入 Fast", "cases": cases, "hints": hints, "rejections": rejections, "hint_rejection": hint_rejection, "old_guard_paths": old_paths, "totals": totals, "checks": checks, "go_candidates": all(checks.values())}


def _hint_rejection() -> dict:
    source = """
def sample(X, Y, N: tl.constexpr, B: tl.constexpr):
    idx = tl.program_id(0) * B + tl.arange(0, B)
    hinted = tl.multiple_of(idx, 16)
    mask = hinted < N
    value = tl.load(X + hinted, mask=mask, other=0)
    tl.store(Y + hinted, value + 1, mask=mask)
"""
    meaning = SEMANTICS["B"]
    ir = parse_access_ir(source, {"X": "X", "Y": "OUT"}, {"X": 4, "Y": 4})
    call = CallBinding((("X", "X"), ("Y", "OUT")), (("N", "x.size(0)"), ("B", "128")), "ceil(N/B)", ("N",), 4, "离线提示拒绝样例")
    site = audit_call_site(run_fast, KERNELS["B"].fn.__name__, call)
    result = extract_alignment(ir, meaning, call, site)
    return {"ir_status": ir.status, "status": result.status, "reason": result.reason, "candidates": [item.to_dict() for item in result.candidates]}


def _case_hint(name: str) -> dict:
    from scenarios.alignment_fixtures import index_hint, pointer_hint, run_index_hint, run_pointer_hint
    kernel, wrapper = (pointer_hint, run_pointer_hint) if name == "pointer" else (index_hint, run_index_hint)
    meaning = SEMANTICS["B"]
    pointers = {"X": "X", "Y": "OUT"}
    ir = parse_access_ir(kernel, pointers, {"X": 4, "Y": 4})
    call = CallBinding((("X", "X"), ("Y", "OUT")), (("N", "x.size(0)"), ("B", "128")), "ceil(N/B)", ("N",), 4, "scenarios/alignment_fixtures.py")
    site = audit_call_site(wrapper, kernel.fn.__name__, call)
    alignment = extract_alignment(ir, meaning, call, site)
    meta = _sample((128,), (1,), 0 if name == "pointer" else 1, 512 if name == "pointer" else 516, 0x1000 if name == "pointer" else 0x1004)
    tensors = {"X": meta, "OUT": _sample((128,), (1,), 0, 512, 0x2000)}
    return {"ir": ir.to_dict(), "call": call.to_dict(), "call_site": asdict(site), "status": {"alignment": alignment.status}, "reason": {"alignment": alignment.reason}, "candidates": {"alignment": [item.to_dict() for item in alignment.candidates]}, "shadow": {"tensor_metadata": {key: asdict(value) for key, value in tensors.items()}, "scalars": {"N": 128, "B": 128}, "evaluation": {"alignment": [item.evaluate(tensors, {"N": 128, "B": 128}) for item in alignment.candidates]}}}


def _bound_text(data: dict) -> str:
    if data["kind"] == "constant":
        return str(data["value"])
    if data["kind"] == "scalar":
        return data["scalar"]
    symbol = {"add": "+", "sub": "-", "mul": "*"}[data["kind"]]
    return f"({_bound_text(data['parts'][0])}{symbol}{_bound_text(data['parts'][1])})"


def _value_text(data: dict) -> str:
    if data["kind"] == "constant":
        return str(data["value"])
    if data["kind"] == "scalar":
        return data["scalar"]
    if data["kind"] in ("size", "stride"):
        return f"{data['tensor']}.{data['kind']}({data['axis']})"
    return f"{data['tensor']}.{data['kind']}"


def _condition_text(data: dict | None) -> str:
    if data is None:
        return "未降为条件的义务"
    kind = data["kind"]
    if kind == "eq":
        return f"{_value_text(data['left'])} == {_value_text(data['right'])}"
    if kind == "mod_eq":
        return f"{_value_text(data['left'])} % {data['divisor']} == {data['remainder']}"
    if kind == "access_span":
        span = data["span"]
        return f"{span['tensor']}[{_bound_text(span['lower'])}..{_bound_text(span['upper'])}] ∈ storage ({span['element_bytes']} B/elem)"
    if kind in ("and", "or"):
        joiner = " 且 " if kind == "and" else " 或 "
        return "(" + joiner.join(_condition_text(item) for item in data["parts"]) + ")"
    return kind


def _record_text(record: dict) -> str:
    if record["condition"] is not None:
        return _condition_text(record["condition"])
    def expr_text(node: dict) -> str:
        return f"{node['op']}({node['value']})" if node["value"] is not None else f"{node['op']}({', '.join(expr_text(item) for item in node['args'])})"
    return f"义务：{expr_text(record['obligation'])}"


def render_candidate_report(data: dict) -> str:
    totals = data["totals"]
    lines = ["# TritonPact 第三阶段候选契约提取报告", "", f"> 生成时间：{data['generated_at']}  ", f"> 验收：**{'通过' if data['go_candidates'] else '未通过'}**  ", "> 候选仅作离线与影子核对；旧 Guard 未接入。", "", "## 计数与检查", "", f"- 已生成记录（含拒绝义务）：{totals['generated']}；静态充分规则：{totals['statically_proven']}；Unproven：{totals['unproven']}；Unknown：{totals['unknown']}。", "", "| 检查 | 结果 |", "| --- | --- |"]
    lines.extend(f"| {name} | {'通过' if passed else '失败'} |" for name, passed in data["checks"].items())
    lines.extend(["", "## 逐条候选", "", "| 案例 | 类型 | 访存来源 | 条件或义务 | 用途 | 状态 | 影子值 | 推导理由 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"])
    for name, case in (*data["cases"].items(), *data["hints"].items()):
        for kind, records in case["candidates"].items():
            for record, value in zip(records, case["shadow"]["evaluation"][kind]):
                origin = record["origin"]
                lines.append(f"| {name} | {kind} | `{origin['source_file']}:{origin['source_line']}` {origin['access_kind']} `{origin['pointer']}` | `{_record_text(record)}` | {record['purpose']} | {record['evidence']} | {value} | {record['reason']} |")
    for name, item in data["rejections"].items():
        for kind, key in (("shape_stride", "shape_candidates"), ("span", "span_candidates")):
            for record in item[key]:
                origin = record["origin"]
                lines.append(f"| reject:{name} | {kind} | `{origin['source_file']}:{origin['source_line']}` {origin['access_kind']} `{origin['pointer']}` | `{_record_text(record)}` | {record['purpose']} | {record['evidence']} | — | {record['reason']} |")
    for record in data["hint_rejection"]["candidates"]:
        origin = record["origin"]
        lines.append(f"| reject:vector_hint | alignment | `{origin['source_file']}:{origin['source_line']}` {origin['access_kind']} `{origin['pointer']}` | `{_record_text(record)}` | {record['purpose']} | {record['evidence']} | — | {record['reason']} |")
    lines.extend(["", "完整 JSON 保留每个访问点的原始/规范化地址、mask、逻辑索引、调用绑定、支持域和未降级义务。影子值来自报告中列明的代表性元数据，不能代表所有运行输入。", "", "## 旧 Guard 路径对照", "", "| 样例 | 路径 |", "| --- | --- |"])
    lines.extend(f"| {name} | {path} |" for name, path in data["old_guard_paths"].items())
    lines.extend(["", "## 拒绝边界", "", "| 变体 | shape/stride | span | 原因 |", "| --- | --- | --- | --- |"])
    lines.extend(f"| {name} | {item['shape_status']} | {item['span_status']} | {item['shape_reason']}; {item['span_reason']} |" for name, item in data["rejections"].items())
    hint = data["hint_rejection"]
    lines.append(f"| vector_hint | — | — | alignment {hint['status']}：{hint['reason']}；未生成指针对齐条件 |")
    lines.extend(["", "## 证据边界", "", "- `Statically-Proven` 表示在声明的独立语义、调用绑定和受支持 IR 规则下，该候选是充分的条件形式；影子值为当前样例的元数据求值。", "- 隔离数值和旧分派路径以单独的第三阶段 `candidate_regression.md/.json` 为准；有限运行不构成一般性安全证明。", "- 间接索引、动态循环、未知 mask/grid、负步长、未知元素宽度和复杂提示形式仍为 Unknown/Unsupported。", "- 新候选没有接入 Fast、Guard 或新的分派路径。", ""])
    return "\n".join(lines)
