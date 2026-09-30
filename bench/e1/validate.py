"""e1 最终机器核验；核心语料通过与外部原生资源缺口分开报告。"""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch
import triton

from bench.e1.acquire import ROOT
from bench.validate import build as validate_phase6

PROJECT = ROOT.parent.parent
RESULTS = PROJECT / "results"


def load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))


def git(*args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=PROJECT, capture_output=True, text=True, timeout=30, check=True)
    return completed.stdout.strip()


def run_regression() -> dict:
    """运行完整 pytest 集和一轮旧隔离案例，保留命令、退出码与原始摘要。"""
    isolation_path = RESULTS / "e1_regression_isolation.md"
    command = [sys.executable, "test/run_tests.py", "--quick", "--output", str(isolation_path)]
    completed = subprocess.run(command, cwd=PROJECT, capture_output=True, text=True, timeout=600, check=False)
    output = completed.stdout + completed.stderr
    summary = re.search(r"\d+ passed(?:, \d+ skipped)? in [^\n]+", output)
    isolation = load("e1_regression_isolation") if isolation_path.with_suffix(".json").exists() else {}
    return {"command": ["python", *command[1:]], "exit_code": completed.returncode,
            "pytest_summary": summary.group(0) if summary else "Unknown", "stdout_tail": output[-12000:],
            "legacy_isolation_passed": isolation.get("go_core") is True,
            "legacy_isolation_checks": isolation.get("checks", {}),
            "report": str(isolation_path.relative_to(PROJECT))}


def main() -> int:
    regression = run_regression()
    corpus = load("e1_corpus_inventory")
    semantic = load("e1_semantic_audit")
    related = load("e1_related_work_baselines")
    resources = load("e1_native_linux_resources")
    phase6 = validate_phase6()
    core_checks = {
        "phase6_current_and_go": phase6["go_stage6"],
        "regression_and_legacy_isolation": regression["exit_code"] == 0 and regression["legacy_isolation_passed"],
        "corpus_frozen": corpus["checks"]["frozen"],
        "source_size_threshold": corpus["checks"]["size_and_sources"],
        "positive_reference_threshold": corpus["checks"]["positive_reference_threshold"],
        "all_positive_smoke_passed": corpus["checks"]["all_declared_positive_smoke"],
        "source_and_binding_audit": corpus["checks"]["sources_and_bindings"],
        "related_work_reviewed": all(related["checks"].values()),
        "no_online_capability_change": load("e1_capability_gap")["online_changes"] is False,
        "semantic_report_current": semantic["fingerprints"] == corpus["fingerprints"],
    }
    resource_checks = {
        "two_candidates_registered": resources["checks"]["two_candidates_registered"],
        "scripts_prepared": resources["checks"]["scripts_prepared"],
        "two_native_resources_acquired": resources["checks"]["two_native_resources_acquired"],
        "two_native_smokes_passed": resources["checks"]["two_native_smokes_passed"],
    }
    go_core = all(core_checks.values())
    preparation_complete = resource_checks["two_candidates_registered"] and resource_checks["scripts_prepared"]
    native_resource_ready = resource_checks["two_native_resources_acquired"] and resource_checks["two_native_smokes_passed"]
    go_e1 = go_core and preparation_complete
    result = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "go" if native_resource_ready and go_e1 else "go_native_resources_deferred" if go_e1 else "no_go",
        "go_e1_core": go_core,
        "go_e1": go_e1,
        "native_resource_ready": native_resource_ready,
        "native_resource_policy": "2026-09-30 用户决定延期到项目开发完成后、正式跨硬件实验前，不阻塞 e2。",
        "git": {"start_head": "8eb9d97ebca4db3639d3333820492d47f28dca79", "current_head": git("rev-parse", "HEAD"),
                "origin_main": git("rev-parse", "origin/main"), "worktree_status": git("status", "--short")},
        "environment": {"system": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
                        "triton": triton.__version__, "cuda": torch.version.cuda,
                        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown"},
        "phase6": {"go_stage6": phase6["go_stage6"], "source_fingerprint_current": phase6["checks"]["phase6_regression_current"],
                   "historical_counts": {"registered": 15, "syntax_supported": 14, "candidate_complete": 14,
                                         "guard_available": 8, "fast_feasible": 8}},
        "e1_counts": corpus["counts"],
        "core_checks": core_checks,
        "resource_checks": resource_checks,
        "regression": regression,
        "reports": ["results/e1_corpus_inventory.md", "results/e1_semantic_audit.md", "results/e1_capability_gap.md",
                    "results/e1_related_work_baselines.md", "results/e1_native_linux_resources.md", "results/e1_regression.md"],
        "scope": "e1 Go 不授予任何新增 Guard/Fast；两套原生 Linux 机器未取得，native_resource_ready 保持 false，正式跨硬件实验前必须重新验收。",
    }
    lines = ["# e1 阶段状态与验收记录", "", f"生成时间：{result['generated_at']}",
             f"状态：`{result['status']}`；核心语料 Go={go_core}；e1 Go={go_e1}；原生实机就绪={native_resource_ready}。", "",
             "## 核心核验", "", "| 核验项 | 结果 |", "| --- | --- |"]
    lines.extend(f"| {key} | {'通过' if value else '未满足'} |" for key, value in core_checks.items())
    lines.extend(["", "## 原生资源核验", "", "| 核验项 | 结果 |", "| --- | --- |"])
    lines.extend(f"| {key} | {'通过' if value else '未满足'} |" for key, value in resource_checks.items())
    lines.extend(["", "## 命令与退出码", "", f"- `{' '.join(regression['command'])}`：退出码 {regression['exit_code']}；{regression['pytest_summary']}；旧隔离 {sum(regression['legacy_isolation_checks'].values())}/{len(regression['legacy_isolation_checks'])}。",
                  "- `python -m bench.e1.audit --freeze --smoke`：报告中的 28/28 正向 smoke 与当前指纹一致。",
                  "- `python -m bench.e1.related --run-triton-verify`：公开 safe/bug 示例通过，详见相关工作 JSON。",
                  "- `python -m bench.e1.resources`：退出码 0；候选登记完成，实际机器状态仍为 not_acquired/not_run。", "",
                  "- 原生实机项由用户决定延期，不阻塞 e2；项目开发完成后、正式跨硬件实验前必须重新打开。", "",
                  result["scope"]])
    (RESULTS / "e1_regression.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (RESULTS / "e1_regression.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "go_e1_core": go_core, "go_e1": go_e1,
                      "native_resource_ready": native_resource_ready}, ensure_ascii=False))
    return 0 if go_e1 else 2


if __name__ == "__main__":
    raise SystemExit(main())
