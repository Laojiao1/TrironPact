"""e2 最终验收：覆盖门槛、冻结留出、历史回归和证据指纹。"""

from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
import triton

from bench.e1.audit import fingerprint as e1_fingerprint
from bench.e2.coverage import PROJECT, RESULTS
from bench.e2.freeze import FREEZE, current_fingerprint
from bench.validate import build as validate_phase6


def _load(name: str) -> dict:
    return json.loads((RESULTS / f"{name}.json").read_text(encoding="utf-8"))


def _git(*args: str) -> str:
    # porcelain 首列的空格是状态字段，不能用 strip() 删除；这里只移除结尾换行。
    return subprocess.run(["git", *args], cwd=PROJECT, capture_output=True, text=True, timeout=30, check=True).stdout.rstrip("\r\n")


def _run_regression() -> dict:
    output = RESULTS / "e2_regression_isolation.md"
    command = [sys.executable, "test/run_tests.py", "--quick", "--output", str(output)]
    completed = subprocess.run(command, cwd=PROJECT, capture_output=True, text=True, timeout=900, check=False)
    text = completed.stdout + completed.stderr
    summary = re.search(r"\d+ passed(?:, \d+ skipped)? in [^\n]+", text)
    isolation = _load("e2_regression_isolation") if output.with_suffix(".json").exists() else {}
    return {
        "command": ["python", *command[1:]],
        "exit_code": completed.returncode,
        "pytest_summary": summary.group(0) if summary else "Unknown",
        "legacy_isolation_passed": isolation.get("go_core") is True,
        "legacy_isolation_checks": isolation.get("checks", {}),
        "stdout_tail": text[-12000:],
        "report": str(output.relative_to(PROJECT)),
    }


def _unexpected_worktree(status: str) -> list[str]:
    allowed = ("README.md", "bench/e2/", "pact/e2_", "results/e2_", "test/test_e2_")
    paths = []
    for line in status.splitlines():
        path = line[3:].replace("\\", "/")
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if path == "bench/catalog.json":
            eol_only = subprocess.run(
                ["git", "diff", "--ignore-space-at-eol", "--exit-code", "--", path],
                cwd=PROJECT,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            ).returncode == 0
            if eol_only:
                continue
        if not path.startswith(allowed):
            paths.append(path)
    return paths


def main() -> int:
    regression = _run_regression()
    access = _load("e2_access_ir_coverage")
    candidates = _load("e2_candidate_extraction")
    guards = _load("e2_guard_coverage")
    e1 = _load("e1_semantic_audit")
    phase6 = validate_phase6()
    freeze = json.loads(FREEZE.read_text(encoding="utf-8"))
    status = _git("status", "--short")
    counts = {
        "registered": 40,
        "positive": access["counts"]["total"],
        "supported": access["counts"].get("Supported", 0),
        "unknown": access["counts"].get("Unknown", 0),
        "unsupported": access["counts"].get("Unsupported", 0),
        "candidate_complete": candidates["counts"]["candidate_complete"],
        "guard_available": guards["counts"]["guard_available"],
        "fast_feasible": guards["counts"]["fast_feasible"],
        "integrated": 0,
        "challenge": 12,
    }
    fast_rows = [row for row in guards["rows"] if row["fast_feasible"]]
    checks = {
        "phase6_current_and_go": phase6["go_stage6"],
        "e1_fingerprint_current": e1.get("fingerprints") == e1_fingerprint(),
        "regression_and_legacy_isolation": regression["exit_code"] == 0 and regression["legacy_isolation_passed"],
        "development_rules_frozen_before_holdout": freeze.get("holdout_evaluated") is True and freeze.get("rule_fingerprint") == current_fingerprint(),
        "holdout_not_used_for_rule_development": all(report.get("holdout_used_for_rule_development") is False for report in (access, candidates, guards)),
        "full_positive_denominator_reported": access["mode"] == candidates["mode"] == guards["mode"] == "development_and_frozen_holdout" and counts["positive"] == 28,
        "supported_threshold": counts["supported"] >= 24,
        "candidate_threshold": counts["candidate_complete"] >= 20,
        "guard_threshold": counts["guard_available"] >= 16,
        "fast_threshold": counts["fast_feasible"] >= 12,
        "family_distribution": all(candidates["complete_by_family"].get(name, 0) >= 3 for name in ("elementwise_mapping", "feature_broadcast", "layout_2d", "row_reduction")),
        "source_distribution": len(candidates["complete_by_source"]) >= 4,
        "all_fast_smoke_current": len(fast_rows) == counts["fast_feasible"] and all(row["smoke_current"] for row in fast_rows),
        "unknown_preserved": counts["unknown"] == 1 and any(row["id"] == "flag_slice" and row["status"] == "Unknown" for row in access["rows"]),
        "no_online_or_e3_changes": not _unexpected_worktree(status),
    }
    go_e2 = all(checks.values())
    result = {
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "go" if go_e2 else "no_go",
        "go_e2": go_e2,
        "git": {
            "start_head": freeze["git_head"],
            "current_head": _git("rev-parse", "HEAD"),
            "origin_main": _git("rev-parse", "origin/main"),
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
        "counts": counts,
        "candidate_complete_by_family": candidates["complete_by_family"],
        "candidate_complete_by_source": candidates["complete_by_source"],
        "checks": checks,
        "regression": regression,
        "rejections": [
            {"id": row["id"], "split": row["split"], "status": row["status"], "reason": row["reason"]}
            for row in access["rows"] if row["status"] != "Supported"
        ],
        "reports": [
            "results/e2_access_ir_coverage.md",
            "results/e2_candidate_extraction.md",
            "results/e2_guard_coverage.md",
            "results/e2_regression.md",
        ],
        "scope": "e2 Fast 可行是冻结调用域的离线证据，未注册在线分派；reduction 仅证明访问域，数值正确性来自有限独立参考 smoke。",
        "backlog": ["TTIR 辅助解析", "复杂静态循环", "组合 reduction 数值证明", "图级 Guard"],
    }
    lines = [
        "# e2 阶段状态与验收记录",
        "",
        f"生成时间：{result['generated_at']}",
        f"状态：`{result['status']}`；go_e2={go_e2}。",
        "",
        f"七层计数：{counts}。",
        f"类别分布：{candidates['complete_by_family']}。",
        f"来源分布：{candidates['complete_by_source']}。",
        "",
        "## 核验项",
        "",
        "| 核验 | 结果 |",
        "| --- | --- |",
    ]
    lines.extend(f"| {key} | {'通过' if value else '未满足'} |" for key, value in checks.items())
    lines.extend(
        [
            "",
            "## 拒绝与边界",
            "",
            *[f"- `{row['id']}`：{row['status']}；{row['reason']}。" for row in result["rejections"]],
            f"- {result['scope']}",
            f"- CCF A backlog：{'、'.join(result['backlog'])}。",
            "",
            "## 回归",
            "",
            f"- `{' '.join(regression['command'])}`：退出码 {regression['exit_code']}；{regression['pytest_summary']}；旧隔离 {sum(regression['legacy_isolation_checks'].values())}/{len(regression['legacy_isolation_checks'])}。",
        ]
    )
    (RESULTS / "e2_regression.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (RESULTS / "e2_regression.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "go_e2": go_e2, "counts": counts}, ensure_ascii=False))
    return 0 if go_e2 else 2


if __name__ == "__main__":
    raise SystemExit(main())
