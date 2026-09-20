"""第四阶段隔离 Oracle；与 PoC 的固定案例 Oracle 分开。"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from pact.mutation import MutationRecipe


ROOT = Path(__file__).resolve().parent.parent


def classify_process(recipe: MutationRecipe, code: int, stdout: str, stderr: str) -> dict:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        return {"category": "process_failure_unknown", "detail": {"reason": "worker 未输出唯一 JSON 行"}, "exit_code": code, "stderr": stderr[-2000:], "evidence": "Unknown"}
    try:
        detail = json.loads(lines[0])
    except (json.JSONDecodeError, TypeError):
        return {"category": "process_failure_unknown", "detail": {"reason": "worker JSON 无效"}, "exit_code": code, "stderr": stderr[-2000:], "evidence": "Unknown"}
    if not isinstance(detail, dict) or detail.get("id") != recipe.id or detail.get("fingerprint") != recipe.fingerprint:
        return {"category": "process_failure_unknown", "detail": {"reason": "worker 结果身份或配方指纹不符"}, "exit_code": code, "stderr": stderr[-2000:], "evidence": "Unknown"}
    kind = detail.get("kind")
    if code == 0 and kind in {"correct", "numeric_mismatch", "metadata_mismatch", "preflight_skip"}:
        category = kind
        evidence = "Unknown" if kind == "preflight_skip" else "Empirically-Validated"
    elif code != 0 and kind == "worker_exception":
        message = (str(detail.get("message", "")) + "\n" + stderr).lower()
        if detail.get("stage") == "protocol":
            category = "oracle_setup_error"
        elif "illegal memory access" in message or ("cuda" in message and "out of bounds" in message):
            category = "memory_error_reported"
        elif detail.get("type") in {"CompilationError", "OutOfResources"} or "launch failure" in message:
            category = "compile_or_launch_error"
        else:
            category = "inconclusive"
        evidence = "Unknown"
    else:
        category, evidence = "process_failure_unknown", "Unknown"
    return {"category": category, "detail": detail, "exit_code": code, "stderr": stderr[-2000:], "evidence": evidence}


def run_mutation(recipe: MutationRecipe, *, timeout: int = 90) -> dict:
    if timeout < 1 or timeout > 300:
        raise ValueError("超时预算无效")
    command = [sys.executable, "-m", "scenarios.mutation_worker"]
    try:
        done = subprocess.run(command, cwd=ROOT, input=json.dumps(recipe.to_dict(), ensure_ascii=False), capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        return {"category": "timeout_unknown", "detail": {"id": recipe.id, "fingerprint": recipe.fingerprint}, "exit_code": None, "stderr": "", "evidence": "Unknown"}
    return classify_process(recipe, done.returncode, done.stdout, done.stderr)
