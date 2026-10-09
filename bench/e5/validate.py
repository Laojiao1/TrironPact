"""e5a 一键总核验：重算指纹、生成追踪状态并运行完整快速回归。"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch
import triton

from bench.e1.audit import fingerprint as e1_fingerprint
from bench.e2.freeze import current_fingerprint
from bench.e3.validate import source_fingerprint as e3_source_fingerprint
from bench.e4.validate import source_fingerprint as e4_source_fingerprint
from bench.e5.holdout import build as build_holdout, write as write_holdout
from bench.e5.readiness import build_ccfa, build_ccfb, write as write_readiness
from bench.e5.traceability import build as build_traceability, write as write_traceability


PROJECT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT / "results"
CATALOG = PROJECT / "bench/e1/catalog.json"
E2_FREEZE = PROJECT / "bench/e2/freeze.json"

EVIDENCE_FILES = (
    "bench/e1/catalog.json",
    "bench/e1/sources.json",
    "bench/e2/freeze.json",
    "results/e1_corpus_inventory.json",
    "results/e1_semantic_audit.json",
    "results/e1_semantic_smoke.json",
    "results/e2_access_ir_coverage.json",
    "results/e2_candidate_extraction.json",
    "results/e2_guard_coverage.json",
    "results/e2_regression.json",
    "results/e3_mutation_comparison.json",
    "results/e3_real_risk_cases.json",
    "results/e3_refinement_audit.json",
    "results/e3_smt_audit.json",
    "results/e3_regression.json",
    "results/e4_workload_manifest.json",
    "results/e4_workload_correctness.json",
    "results/e4_dispatch_baselines.json",
    "results/e4_wsl_diagnostics.json",
    "results/e4_cross_hardware_performance.json",
    "results/e4_regression.json",
)


def _load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT, capture_output=True, text=True, timeout=30, check=True).stdout.rstrip("\r\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_fingerprint() -> str:
    """绑定 e5a 实现、说明、测试和作为输入的冻结阶段状态。"""

    paths = [
        *(PROJECT / "bench/e5").glob("*.py"),
        *(PROJECT / "test").glob("test_e5_*.py"),
        PROJECT / "artifact/e5_correctness.md",
        CATALOG,
        E2_FREEZE,
        RESULTS / "e3_regression.json",
        RESULTS / "e4_regression.json",
    ]
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(path.relative_to(PROJECT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _unexpected_worktree(status: str) -> list[str]:
    allowed = ("artifact/", "bench/e5/", "results/e5_", "test/test_e5_")
    unexpected = []
    for line in status.splitlines():
        path = line[3:].replace("\\", "/")
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
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
            unexpected.append(path)
    return unexpected


def _valid_report_time(value: object, now: datetime) -> bool:
    if not isinstance(value, str):
        return False
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return False
    if parsed.tzinfo is None:
        return False
    return parsed.astimezone(timezone.utc) <= now.astimezone(timezone.utc) + timedelta(minutes=5)


def build_evidence_manifest() -> dict:
    """对正式输入资产做内容寻址；缺失或无效 JSON 直接保留失败。"""

    rows = []
    now = datetime.now().astimezone()
    for relative in EVIDENCE_FILES:
        path = PROJECT / relative
        row = {"path": relative, "exists": path.is_file(), "sha256": None, "json_valid": None, "time_valid": None}
        if path.is_file():
            row["sha256"] = _sha256(path)
            if path.suffix == ".json":
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    row["json_valid"] = False
                else:
                    row["json_valid"] = True
                    row["time_valid"] = True if "generated_at" not in data else _valid_report_time(data["generated_at"], now)
        rows.append(row)
    return {
        "generated_at": now.isoformat(),
        "git_head": _git("rev-parse", "HEAD"),
        "branch": _git("branch", "--show-current"),
        "rows": rows,
        "checks": {
            "all_files_present": all(row["exists"] for row in rows),
            "all_json_valid": all(row["json_valid"] is not False for row in rows),
            "all_report_times_valid": all(row["time_valid"] is not False for row in rows),
        },
    }


def _write_manifest(report: dict) -> None:
    base = RESULTS / "e5_evidence_manifest"
    base.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# e5a 正式证据内容清单",
        "",
        f"- Git：`{report['git_head']}`（`{report['branch']}`）。",
        "- SHA-256 绑定既有正式 JSON、冻结清单与原始数据；它不改变历史报告。",
        "",
        "| 路径 | SHA-256 | JSON | 时间 |",
        "| --- | --- | --- | --- |",
        *[f"| `{row['path']}` | `{row['sha256'] or 'missing'}` | {row['json_valid']} | {row['time_valid']} |" for row in report["rows"]],
        "",
    ]
    base.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def _run_regression() -> dict:
    output = RESULTS / "e5_regression_isolation.md"
    command = [sys.executable, "test/run_tests.py", "--quick", "--output", str(output)]
    done = subprocess.run(command, cwd=PROJECT, capture_output=True, text=True, timeout=1500, check=False)
    text = done.stdout + done.stderr
    summary = re.search(r"\d+ passed(?:, \d+ skipped)? in [^\n]+", text)
    isolation = _load("e5_regression_isolation") if output.with_suffix(".json").exists() else {}
    return {
        "command": ["python", *command[1:]],
        "exit_code": done.returncode,
        "pytest_summary": summary.group(0) if summary else "Unknown",
        "legacy_isolation_passed": isolation.get("go_core") is True,
        "legacy_checks": isolation.get("checks", {}),
        "stdout_tail": text[-12000:],
        "report": str(output.relative_to(PROJECT)).replace("\\", "/"),
    }


def _split_check() -> bool:
    catalog = json.loads(CATALOG.read_text(encoding="utf-8"))
    groups = {name: {row["id"] for row in catalog["entries"] if row["split"] == name} for name in ("development", "holdout", "challenge")}
    return not (groups["development"] & groups["holdout"] or groups["development"] & groups["challenge"] or groups["holdout"] & groups["challenge"])


def build(regression: dict, manifest: dict) -> dict:
    e1 = _load("e1_semantic_audit")
    e2 = _load("e2_regression")
    freeze = json.loads(E2_FREEZE.read_text(encoding="utf-8"))
    e3 = _load("e3_regression")
    risk = _load("e3_real_risk_cases")
    mutation = _load("e3_mutation_comparison")
    smt = _load("e3_smt_audit")
    e4 = _load("e4_regression")
    correctness = _load("e4_workload_correctness")
    baselines = _load("e4_dispatch_baselines")
    diagnostics = _load("e4_wsl_diagnostics")
    cross = _load("e4_cross_hardware_performance")
    holdout = _load("e5_holdout_evaluation")
    trace = _load("e5_traceability")
    ccfb = _load("e5_ccfb_readiness")
    ccfa = _load("e5_ccfa_gap")
    status = _git("status", "--short")
    counts = e2["counts"]
    environment_tuple = lambda report: tuple(report.get("environment", {}).get(key) for key in ("python", "torch", "triton", "cuda", "gpu"))
    checks = {
        "git_head_matches_origin_eval": _git("rev-parse", "HEAD") == _git("rev-parse", "origin/eval"),
        "e1_corpus_and_source_fingerprints_current": e1.get("fingerprints") == e1_fingerprint(),
        "e2_rule_fingerprint_current": e2.get("go_e2") is True and freeze.get("rule_fingerprint") == current_fingerprint(),
        "e3_source_fingerprint_current": e3.get("go_e3") is True and e3.get("source_fingerprint") == e3_source_fingerprint(),
        "e4a_source_fingerprint_current": e4.get("go_e4_integration") is True and e4.get("source_fingerprint") == e4_source_fingerprint(),
        "formal_evidence_manifest_complete": all(manifest["checks"].values()),
        "environment_fingerprints_consistent": len({environment_tuple(report) for report in (e2, e3, e4)}) == 1,
        "dataset_splits_disjoint": _split_check(),
        "seven_layer_denominators_preserved": counts == {
            "registered": 40, "positive": 28, "supported": 27, "unknown": 1, "unsupported": 0,
            "candidate_complete": 27, "guard_available": 27, "fast_feasible": 27, "integrated": 0, "challenge": 12,
        },
        "holdout_read_only_audit_complete": holdout.get("go_holdout_audit") is True and holdout.get("rerun_performed") is False,
        "zero_known_false_allows": holdout["counts"]["known_false_allows"] == 0 and correctness["counts"]["known_false_allows"] == 0,
        "real_risk_layers_and_minimum_complete": risk.get("go_risk") is True and risk["counts"]["l1_witnesses"] >= 3 and len(risk["counts"]["by_source"]) >= 2,
        "fair_system_baselines_complete": baselines.get("go_baselines") is True and all(baselines.get("checks", {}).values()),
        "secondary_results_and_diagnostics_disclosed": mutation.get("go_mutation") is True
        and smt["counts"]["real_candidate_deletions"] == 1
        and smt["counts"]["synthetic_deletions"] == 0
        and diagnostics["checks"]["diagnostic_scope_only"],
        "correctness_argument_and_traceability_complete": (PROJECT / "artifact/e5_correctness.md").is_file() and trace.get("go_traceability") is True,
        "readiness_state_consistent": ccfb.get("go_e5_ccfb_readiness") is all(ccfb.get("conditions", {}).values())
        and (cross.get("go_e4_cross_hardware") is True or ccfb.get("go_e5_ccfb_readiness") is False)
        and ccfa.get("go_e5_ccfa") is False,
        "only_e5a_worktree_changes": not _unexpected_worktree(status),
        "full_tests_and_legacy_isolation": regression["exit_code"] == 0 and regression["legacy_isolation_passed"],
    }
    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "preparation_go_final_readiness_pending" if all(checks.values()) else "no_go",
        "go_e5_preparation": all(checks.values()),
        "go_e5_ccfb_readiness": ccfb["go_e5_ccfb_readiness"],
        "go_e5": ccfb["go_e5_ccfb_readiness"],
        "source_fingerprint": source_fingerprint(),
        "git": {
            "current_head": _git("rev-parse", "HEAD"),
            "origin_eval": _git("rev-parse", "origin/eval"),
            "branch": _git("branch", "--show-current"),
            "worktree_status": status,
            "unexpected_paths": _unexpected_worktree(status),
        },
        "environment": {
            "system": platform.platform(), "python": platform.python_version(), "torch": torch.__version__,
            "triton": triton.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "Unknown",
        },
        "checks": checks,
        "regression": regression,
        "pending": [row for row in ccfb["rows"] if not row["passed"]],
        "claims": {
            "preparation": "e5a 的支持域、证明草图、追踪、只读留出审计和总核验代码已形成。",
            "readiness": "原生 Linux 双硬件、第二审阅人、提交后干净 checkout 与最终提交绑定尚未完成。",
            "holdout": "e5a 没有重跑留出 Kernel，也没有修改 e2 冻结规则。",
        },
    }


def write(report: dict) -> None:
    (RESULTS / "e5_regression.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# e5a 阶段状态与统一核验",
        "",
        f"- 状态：`{report['status']}`。",
        f"- `go_e5_preparation={str(report['go_e5_preparation']).lower()}`；`go_e5_ccfb_readiness={str(report['go_e5_ccfb_readiness']).lower()}`；`go_e5=false`。",
        f"- 源码指纹：`{report['source_fingerprint']}`。",
        "",
        "## 准备项核验",
        "",
        "| 核验 | 结果 |",
        "| --- | --- |",
        *[f"| {key} | {'通过' if value else '未满足'} |" for key, value in report["checks"].items()],
        "",
        "## 最终 readiness 未满足项",
        "",
        *[f"- `{row['id']}`：{row['reason']}" for row in report["pending"]],
        "",
        "## 结论边界",
        "",
        *[f"- **{key}**：{value}" for key, value in report["claims"].items()],
        "",
        "## 回归",
        "",
        f"- `{' '.join(report['regression']['command'])}`：退出码 {report['regression']['exit_code']}；{report['regression']['pytest_summary']}；旧隔离 {sum(report['regression']['legacy_checks'].values())}/{len(report['regression']['legacy_checks'])}。",
        "",
    ]
    (RESULTS / "e5_regression.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    argparse.ArgumentParser().parse_args()
    initial_worktree_clean = _git("status", "--short") == ""
    manifest = build_evidence_manifest()
    _write_manifest(manifest)
    holdout = build_holdout()
    write_holdout(holdout)
    trace = build_traceability()
    write_traceability(trace)
    ccfb, ccfa = build_ccfb(), build_ccfa()
    write_readiness(ccfb, ccfa)
    regression = _run_regression()
    ccfb = build_ccfb(
        verification_context={
            "initial_worktree_clean": initial_worktree_clean,
            "one_click_checks_passed": regression["exit_code"] == 0 and regression["legacy_isolation_passed"],
        }
    )
    write_readiness(ccfb, ccfa)
    report = build(regression, manifest)
    write(report)
    print(json.dumps({"status": report["status"], "go_e5_preparation": report["go_e5_preparation"], "go_e5_ccfb_readiness": report["go_e5_ccfb_readiness"], "go_e5": report["go_e5"]}, ensure_ascii=False))
    return 0 if report["go_e5_preparation"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

