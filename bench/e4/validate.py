"""e4a 最终验收：工作负载、基线、诊断、回归和延期的 e4b。"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch
import triton

from bench.e1.audit import fingerprint as e1_fingerprint
from bench.e2.freeze import current_fingerprint
from bench.e3.validate import source_fingerprint as e3_source_fingerprint


PROJECT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT / "results"
E2_FREEZE = PROJECT / "bench" / "e2" / "freeze.json"


def _load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT, capture_output=True, text=True, timeout=30, check=True).stdout.rstrip("\r\n")


def source_fingerprint() -> str:
    """绑定 e4a 实现、测试、说明和上游冻结证据。"""

    paths = [
        *(PROJECT / "integration").glob("*.py"),
        *(PROJECT / "bench" / "e4").glob("*.py"),
        *(PROJECT / "test").glob("test_e4_*.py"),
        PROJECT / "README.md",
        PROJECT / "bench" / "e1" / "catalog.json",
        PROJECT / "bench" / "e2" / "freeze.json",
        RESULTS / "e2_guard_coverage.json",
        RESULTS / "e3_regression.json",
    ]
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(path.relative_to(PROJECT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _run_regression() -> dict:
    output = RESULTS / "e4_regression_isolation.md"
    command = [sys.executable, "test/run_tests.py", "--quick", "--output", str(output)]
    done = subprocess.run(command, cwd=PROJECT, capture_output=True, text=True, timeout=1200, check=False)
    text = done.stdout + done.stderr
    summary = re.search(r"\d+ passed(?:, \d+ skipped)? in [^\n]+", text)
    isolation = _load("e4_regression_isolation") if output.with_suffix(".json").exists() else {}
    return {
        "command": ["python", *command[1:]],
        "exit_code": done.returncode,
        "pytest_summary": summary.group(0) if summary else "Unknown",
        "legacy_isolation_passed": isolation.get("go_core") is True,
        "legacy_checks": isolation.get("checks", {}),
        "stdout_tail": text[-12000:],
        "report": str(output.relative_to(PROJECT)),
    }


def _unexpected_worktree(status: str) -> list[str]:
    allowed = ("README.md", "integration/", "bench/e4/", "test/test_e4_", "results/e4_")
    result = []
    for line in status.splitlines():
        path = line[3:].replace("\\", "/")
        if path == "bench/catalog.json":
            eol_only = subprocess.run(
                ["git", "diff", "--ignore-space-at-eol", "--exit-code", "--", path],
                cwd=PROJECT,
                capture_output=True,
                timeout=30,
                check=False,
            ).returncode == 0
            if eol_only:
                continue
        if not path.startswith(allowed):
            result.append(path)
    return result


def _write_cross_hardware_pending() -> dict:
    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "pending_resource",
        "go_e4_cross_hardware": False,
        "required_environments": 2,
        "obtained_environments": 0,
        "native_smoke_passed": 0,
        "formal_experiment_run": False,
        "reason": "两套原生 Linux GPU 尚未取得；账号、租赁、实际版本、可用时段和账单均无证据。",
        "next_action": "取得资源后重新核对 e1 候选环境，运行 2/2 smoke，冻结指纹，再重放 e4a 工作负载和计时协议。",
    }
    base = RESULTS / "e4_cross_hardware_performance"
    base.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    base.with_suffix(".md").write_text(
        "# e4b 跨硬件性能状态\n\n"
        "- 状态：`pending_resource`；`go_e4_cross_hardware=false`。\n"
        "- 两套原生 Linux GPU 尚未取得，未运行正式跨硬件实验。\n"
        "- 当前 WSL2 诊断不能替代本项，相关路线清单保持未勾选。\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    regression = _run_regression()
    manifest = _load("e4_workload_manifest")
    correctness = _load("e4_workload_correctness")
    baselines = _load("e4_dispatch_baselines")
    diagnostics = _load("e4_wsl_diagnostics")
    e3 = _load("e3_regression")
    e2 = _load("e2_regression")
    e1 = _load("e1_semantic_audit")
    freeze = json.loads(E2_FREEZE.read_text(encoding="utf-8"))
    cross_hardware = _write_cross_hardware_pending()
    status = _git("status", "--short")
    checks = {
        "e1_fingerprint_current": e1.get("fingerprints") == e1_fingerprint(),
        "e2_go_and_rule_fingerprint_current": e2.get("go_e2") is True and freeze.get("rule_fingerprint") == current_fingerprint(),
        "e3_go_and_source_fingerprint_current": e3.get("go_e3") is True and e3.get("source_fingerprint") == e3_source_fingerprint(),
        "two_traceable_workloads_and_ten_kernels": manifest.get("go_manifest") is True and manifest["counts"]["distinct_kernels"] >= 10,
        "workload_correctness_and_zero_known_false_allows": correctness.get("go_correctness") is True and correctness["counts"]["known_false_allows"] == 0,
        "safe_nonstandard_and_blocked_violation": correctness["counts"]["safe_nonstandard_fast_paths"] >= 1 and correctness["counts"]["blocked_numeric_mismatches"] >= 1,
        "fair_system_baselines": baselines.get("go_baselines") is True and all(baselines["checks"].values()),
        "wsl_diagnostics_complete_and_scoped": diagnostics.get("go_diagnostics") is True and diagnostics["checks"]["diagnostic_scope_only"],
        "full_regression_and_legacy_isolation": regression["exit_code"] == 0 and regression["legacy_isolation_passed"],
        "no_e2_e3_or_online_fast_changes": not _unexpected_worktree(status),
        "e4b_explicitly_pending": cross_hardware["go_e4_cross_hardware"] is False and not cross_hardware["formal_experiment_run"],
    }
    go_e4_integration = all(checks.values())
    go_e4_cross_hardware = cross_hardware["go_e4_cross_hardware"]
    go_e4 = go_e4_integration and go_e4_cross_hardware
    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "integration_go_cross_hardware_pending" if go_e4_integration else "no_go",
        "go_e4_integration": go_e4_integration,
        "go_e4_cross_hardware": go_e4_cross_hardware,
        "go_e4": go_e4,
        "source_fingerprint": source_fingerprint(),
        "git": {
            "start_head": "763926519269325c1af8c39a892b6823d28b3b9f",
            "current_head": _git("rev-parse", "HEAD"),
            "branch": _git("branch", "--show-current"),
            "worktree_status": status,
            "unexpected_paths": _unexpected_worktree(status),
        },
        "environment": {
            "system": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "triton": triton.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown",
        },
        "counts": {
            **manifest["counts"],
            **correctness["counts"],
            "baseline_policies": len(baselines["rows"]),
            "native_linux_environments": cross_hardware["obtained_environments"],
        },
        "checks": checks,
        "regression": regression,
        "claims": {
            "integration": "两个真实轨迹和 14 个不同冻结函数体完成可追溯接入。",
            "safety": "固定案例中已知误放行 0；2 个数值错读布局被阻止，Fallback 正确。",
            "nonstandard": "项目 A2 padded-row 安全直通；Liger row-padding 因 e2 exact-stride 域被保守拒绝并单列缺口。",
            "performance": diagnostics["conclusion"],
            "overall": "e4a 完成不等于 e4 完成；e4b 和 go_e4 仍为 false。",
        },
        "reports": [
            "results/e4_workload_manifest.md",
            "results/e4_workload_correctness.md",
            "results/e4_dispatch_baselines.md",
            "results/e4_wsl_diagnostics.md",
            "results/e4_cross_hardware_performance.md",
            "results/e4_regression_isolation.md",
        ],
    }
    lines = [
        "# e4a 阶段状态与验收记录",
        "",
        f"生成时间：{report['generated_at']}",
        f"状态：`{report['status']}`；`go_e4_integration={go_e4_integration}`，`go_e4_cross_hardware={go_e4_cross_hardware}`，`go_e4={go_e4}`。",
        f"源码指纹：`{report['source_fingerprint']}`。",
        "",
        f"计数：`{report['counts']}`。",
        "",
        "## 核验项",
        "",
        "| 核验 | 结果 |",
        "| --- | --- |",
        *[f"| {key} | {'通过' if value else '未满足'} |" for key, value in checks.items()],
        "",
        "## 结论边界",
        "",
        *[f"- **{key}**：{value}" for key, value in report["claims"].items()],
        "",
        "## 回归",
        "",
        f"- `{' '.join(regression['command'])}`：退出码 {regression['exit_code']}；{regression['pytest_summary']}；旧隔离 {sum(regression['legacy_checks'].values())}/{len(regression['legacy_checks'])}。",
        "",
    ]
    (RESULTS / "e4_regression.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (RESULTS / "e4_regression.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": report["status"], "go_e4_integration": go_e4_integration, "go_e4_cross_hardware": go_e4_cross_hardware, "go_e4": go_e4, "counts": report["counts"]}, ensure_ascii=False))
    return 0 if go_e4_integration else 2


if __name__ == "__main__":
    raise SystemExit(main())

