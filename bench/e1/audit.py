"""e1 来源、语义、能力缺口和隔离数值报告；保留所有拒绝和失败。"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
import triton

from bench.e1.acquire import ROOT
from bench.e1.catalog import CATALOG, load, sha256
from bench.e1.source import extract
from pact.access_ir import parse_access_ir

PROJECT = ROOT.parent.parent
RESULTS = PROJECT / "results"


def fingerprint() -> dict[str, str]:
    """分别绑定清单、原始来源账本与审计实现，历史阶段指纹不受影响。"""
    paths = sorted(ROOT.glob("*.py")) + sorted((PROJECT / "test").glob("test_e1_*.py"))
    raw = b"".join(str(p.relative_to(PROJECT)).encode() + p.read_bytes() for p in paths)
    return {"corpus": sha256(CATALOG.read_bytes()), "sources": sha256((ROOT / "sources.json").read_bytes()), "implementation": sha256(raw)}


def save(name: str, data: dict, markdown: str) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (RESULTS / f"{name}.md").write_text(markdown + "\n", encoding="utf-8")


def probe(row: dict) -> dict:
    """按原始片段摸底；原始 dtype 宽度逐参数登记，Unknown 不授予 Fast。

    AST 本身只解释有限访存语法。首个拒绝原因原样保存，若解析器未
    提供行号则保留 reason 并明确 node_location_unknown，不猜行号。
    """
    fn = extract(ROOT / "upstream" / row["source"] / row["file"], row["function"])
    widths = {}
    if row["smoke"]:
        dtype = {t["name"]: t["dtype"] for t in row["smoke"]["tensors"]}
        widths = {p: 1 if dtype[t] == "bool" else 4 for p,t in row["pointers"].items()}
    parsed = parse_access_ir(fn.source, row["pointers"], widths)
    return {"id": row["id"], "status": parsed.status, "reason": parsed.reason,
            "first_rejection": None if parsed.status == "Supported" else {"parser_reason": parsed.reason, "source_file": row["file"], "function_line": row["line"], "node_location": "参见 parser_reason；无行号时 Unknown"},
            "accesses_before_rejection": len(parsed.accesses)}


def run_smoke(data: dict) -> dict:
    """每个正向样本独立子进程，超时/非零/协议错误不算成功。

    每次完成后保存进度，允许报告中保留部分完成证据；正式结论要求
    全部登记 ID 和两种种子完整，并核对当前源码与语料指纹。
    """
    report = {"generated_at": datetime.now().astimezone().isoformat(), "fingerprints": fingerprint(),
              "environment": {"system": platform.platform(), "python": platform.python_version(), "torch": torch.__version__, "triton": triton.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)},
              "rows": [], "scope": "冻结输入参考 smoke；非 Guard/契约恢复评估，holdout 未用于规则调试"}
    for row in data["entries"]:
        if row["smoke"] is None:
            continue
        command = [sys.executable, "-m", "bench.e1.worker", row["id"]]
        started = time.monotonic()
        try:
            done = subprocess.run(command, cwd=PROJECT, capture_output=True, text=True, timeout=120, check=False)
        except subprocess.TimeoutExpired as error:
            result = {"id": row["id"], "status": "Unknown", "reason": "worker_timeout", "exit_code": None, "stderr": str(error)}
        else:
            try:
                detail = json.loads(done.stdout.splitlines()[-1])
            except (json.JSONDecodeError, IndexError):
                detail = {"id": row["id"], "status": "Unknown", "reason": "worker_exception_or_invalid_output", "stdout": done.stdout[-2000:]}
            result = {**detail, "exit_code": done.returncode, "stderr": done.stderr[-6000:]}
            if done.returncode != 0 or detail.get("id") != row["id"]:
                result["status"] = "failed" if detail.get("status") == "failed" else "Unknown"
        result["command"] = ["python", *command[1:]]
        result["elapsed_seconds"] = time.monotonic() - started
        report["rows"].append(result)
        save("e1_semantic_smoke", report, "# e1 隔离参考 smoke\n\n" + "\n".join(f"- {r['id']}：{r['status']}；退出码 {r['exit_code']}。" for r in report["rows"]))
        print(row["id"], result["status"], "exit=", result["exit_code"], flush=True)
    return report


def report(data: dict, smoke: dict | None) -> dict:
    """重新审计当前清单；过期 smoke 不能充作语义完整证据。"""
    entries = data["entries"]
    probes = [probe(r) for r in entries]
    positive = [r for r in entries if r["smoke"]]
    current = bool(smoke and smoke.get("fingerprints") == fingerprint())
    passed = {r["id"] for r in smoke["rows"] if r["status"] == "passed" and r["exit_code"] == 0} if current else set()
    families = Counter(r["family"] for r in positive if r["id"] in passed)
    counts = {"registered": len(entries), "external": len(entries), "sources": len({r["source"] for r in entries}),
              "syntax": dict(Counter(p["status"] for p in probes)), "positive_declared": len(positive), "semantic_complete": len(passed),
              "positive_by_family": dict(families), "splits": dict(Counter(r["split"] for r in entries)),
              "candidate_complete": 0, "guard_available": 0, "fast_feasible": 0, "integrated": 0}
    result = {"generated_at": datetime.now().astimezone().isoformat(), "fingerprints": fingerprint(), "counts": counts, "smoke_current": current,
              "checks": {"sources_and_bindings": True, "size_and_sources": len(entries) >= 40 and len({r["source"] for r in entries}) >= 4,
                         "positive_reference_threshold": len(passed) >= 28 and all(families.get(f,0) >= 5 for f in ("elementwise_mapping","layout_2d","feature_broadcast","row_reduction")),
                         "all_declared_positive_smoke": len(passed) == len(positive), "frozen": data["freeze_status"] == "frozen"},
              "entries": [{"id":r["id"],"source":r["source"],"family":r["family"],"split":r["split"],"semantic_status":"complete_in_smoke_domain" if r["id"] in passed else "Unknown" if r["smoke"] else "not_required_challenge",**p} for r,p in zip(entries,probes)],
              "scope":"仅 e1 语料与有界语义证据。语法、语义完整、候选、Guard、Fast、接入分母分列；旧15个历史样本另存，未混入这40个真实样本。"}
    inventory = "# e1 真实语料清单\n\n" + json.dumps(counts,ensure_ascii=False,indent=2) + "\n\n| ID | 来源 | 类别 | 集合 | AST | 语义 |\n| --- | --- | --- | --- | --- | --- |\n"
    inventory += "\n".join(f"| {r['id']} | {r['source']} | {r['family']} | {r['split']} | {r['status']} | {r['semantic_status']} |" for r in result["entries"])
    save("e1_corpus_inventory", result, inventory + "\n\n" + result["scope"])
    gap = Counter(feature for row in entries if row["split"] == "development" for feature,value in row["features"].items() if value and feature not in {"load_count","store_count","program_id_axes","explicit_masks"})
    gap_rows = [{"id": row["id"], "source": row["source"], "family": row["family"], "split": row["split"],
                 "features": row["features"], **probe_result} for row, probe_result in zip(entries, probes)]
    gap_data = {"generated_at": result["generated_at"], "fingerprints": fingerprint(), "development_features": dict(gap.most_common()),
                "by_status": dict(Counter(p["status"] for p in probes)),
                "by_family": {family: dict(Counter(p["status"] for row,p in zip(entries,probes) if row["family"] == family)) for family in sorted({r["family"] for r in entries})},
                "all_probes":gap_rows,
                "holdout_policy":"摸底结果归档但不用于规则设计排序；e2 只读取 development 的需求，留出集正式恢复评估在规则冻结后进行", "online_changes":False}
    save("e1_capability_gap", gap_data, "# e1 能力缺口\n\n开发集语法频次（不代表规则已实现）：\n\n" + "\n".join(f"- {k}：{v} 个 Kernel。" for k,v in gap.most_common()) + "\n\n| Kernel | 来源 | 类别 | 集合 | AST | load/store | grid 轴 | mask | 首个拒绝原因 |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n" + "\n".join(f"| {r['id']} | {r['source']} | {r['family']} | {r['split']} | {r['status']} | {r['features']['load_count']}/{r['features']['store_count']} | {r['features']['program_id_axes']} | {r['features']['explicit_masks']} | {r['reason']} |" for r in gap_rows) + "\n\n" + gap_data["holdout_policy"])
    semantic = {**result,"audit_level":"单人源码/语义核对 + 两种冻结输入隔离核对；第二审阅者未完成", "reference_reuse":"Liger 保存黄金函数；Triton/PyTorch/Unsloth 保存正式测试的公式和构造；本地参考改写逐例记录；sub/masked_add 为独立 Eager 规格，不冒称上游黄金测试", "lineage":"Liger RMSNorm 明确继承 Unsloth；两者是不同函数体，但不视为独立算法来源。PyTorch add 的同构副本未重复登记。"}
    save("e1_semantic_audit",semantic,"# e1 来源、语义与绑定审计\n\n"+json.dumps(result["checks"],ensure_ascii=False,indent=2)+"\n\n"+semantic["audit_level"]+"\n\n"+semantic["reference_reuse"]+"\n\n"+semantic["lineage"]+"\n\n完整逐参数、输入、输出、grid、参考、适配和来源哈希见 bench/e1/catalog.json；逐输入结果与 stderr 见 e1_semantic_smoke.json。")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    data = load()
    if args.freeze:
        data["freeze_status"] = "frozen"
        CATALOG.write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    smoke_path = RESULTS / "e1_semantic_smoke.json"
    smoke = run_smoke(data) if args.smoke else json.loads(smoke_path.read_text()) if smoke_path.exists() else None
    status = report(data, smoke)
    print(json.dumps(status["checks"], ensure_ascii=False))
    return 0 if all(status["checks"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
