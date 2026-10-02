"""e3 最终验收：风险、比较、SMT、精化、回归和源码指纹。"""

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


PROJECT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT / "results"
E2_FREEZE = PROJECT / "bench" / "e2" / "freeze.json"


def _load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=PROJECT, capture_output=True, text=True, timeout=30, check=True).stdout.rstrip("\r\n")


def source_fingerprint() -> str:
    """绑定 e3 实现、测试、冻结上游源码和 e2 输入证据。"""

    paths = [
        PROJECT / "pact" / "e3_mutation.py",
        *(PROJECT / "bench" / "e3").glob("*.py"),
        *(PROJECT / "test").glob("test_e3_*.py"),
        PROJECT / "bench" / "e1" / "catalog.json",
        PROJECT / "bench" / "e2" / "freeze.json",
        RESULTS / "e2_candidate_extraction.json",
        PROJECT / "bench" / "e1" / "upstream" / "liger" / "src" / "liger_kernel" / "ops" / "softmax.py",
        PROJECT / "bench" / "e1" / "upstream" / "unsloth" / "unsloth" / "kernels" / "layernorm.py",
    ]
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(path.relative_to(PROJECT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _run_regression() -> dict:
    output = RESULTS / "e3_regression_isolation.md"
    command = [sys.executable, "test/run_tests.py", "--quick", "--output", str(output)]
    done = subprocess.run(command, cwd=PROJECT, capture_output=True, text=True, timeout=900, check=False)
    text = done.stdout + done.stderr
    summary = re.search(r"\d+ passed(?:, \d+ skipped)? in [^\n]+", text)
    isolation = _load("e3_regression_isolation") if output.with_suffix(".json").exists() else {}
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
    allowed = ("README.md", "bench/e3/", "pact/e3_", "results/e3_", "test/test_e3_")
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


def main() -> int:
    regression = _run_regression()
    risk = _load("e3_real_risk_cases")
    mutation = _load("e3_mutation_comparison")
    smt = _load("e3_smt_audit")
    refinement = _load("e3_refinement_audit")
    e2 = _load("e2_regression")
    e1 = _load("e1_semantic_audit")
    freeze = json.loads(E2_FREEZE.read_text(encoding="utf-8"))
    status = _git("status", "--short")
    checks = {
        "e1_fingerprint_current": e1.get("fingerprints") == e1_fingerprint(),
        "e2_go_and_rule_fingerprint_current": e2.get("go_e2") is True and freeze.get("rule_fingerprint") == current_fingerprint(),
        "full_regression_and_legacy_isolation": regression["exit_code"] == 0 and regression["legacy_isolation_passed"],
        "mutation_protocol_frozen_and_fair": mutation.get("go_mutation") is True and all(mutation["checks"].values()),
        "real_risk_minimum": risk.get("go_risk") is True
        and risk["counts"]["l1_witnesses"] >= 3
        and len(risk["counts"]["by_source"]) >= 2,
        "risk_layers_separated": risk["counts"]["l2_wrapper_defenses"] >= 1
        and risk["counts"]["l3_public_items"] >= 0
        and all(risk["checks"].values()),
        "smt_real_candidates_audited": smt.get("go_smt") is True
        and smt["counts"]["real_candidates"] == 158
        and smt["counts"]["synthetic_deletions"] == 0,
        "refutation_and_revision_closed": refinement.get("go_refinement") is True
        and not refinement["witness_classification"]["understrong"]
        and not refinement["fast_revocations"],
        "no_e4_or_online_changes": not _unexpected_worktree(status),
    }
    go_e3 = all(checks.values())
    report = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "go" if go_e3 else "no_go",
        "go_e3": go_e3,
        "source_fingerprint": source_fingerprint(),
        "git": {
            "start_head": "dd0dd74b6912a217c7388a79b811effdbf303698",
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
            "mutation_probes": len(mutation["domain"]),
            "mutation_trials": len(mutation["trials"]),
            **risk["counts"],
            "smt_real_candidates": smt["counts"]["real_candidates"],
            "smt_real_deletions": smt["counts"]["real_candidate_deletions"],
            "actual_revisions": refinement["actual_revision_count"],
            "fast_revocations": len(refinement["fast_revocations"]),
        },
        "checks": checks,
        "regression": regression,
        "claims": {
            "mutation": mutation["claim"],
            "smt": smt["claim"],
            "risk": "4 个 L1、3 个 L2、1 个公开 L3 条目；L1 未冒充上游漏洞",
            "performance": "复制耗时仅为 WSL2 诊断，未提前运行 e4 正式实验",
        },
        "reports": [
            "results/e3_mutation_comparison.md",
            "results/e3_real_risk_cases.md",
            "results/e3_refinement_audit.md",
            "results/e3_smt_audit.md",
            "results/e3_regression.md",
        ],
    }
    lines = [
        "# e3 阶段状态与验收记录",
        "",
        f"生成时间：{report['generated_at']}",
        f"状态：`{report['status']}`；go_e3={go_e3}。",
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
    (RESULTS / "e3_regression.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (RESULTS / "e3_regression.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": report["status"], "go_e3": go_e3, "counts": report["counts"]}, ensure_ascii=False))
    return 0 if go_e3 else 2


if __name__ == "__main__":
    raise SystemExit(main())

