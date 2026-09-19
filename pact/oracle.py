"""启动独立子进程执行用例，防止非法显存访问污染主进程 CUDA Context"""

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OracleResult:
    category: str
    detail: dict
    exit_code: int | None
    stderr: str
    evidence: dict

    def to_dict(self) -> dict:
        return {"category": self.category, "detail": self.detail, "exit_code": self.exit_code, "stderr": self.stderr, "evidence": self.evidence}


def _evidence(category: str, detail: dict) -> dict:
    """静态放行依据与单次运行观察分开记，避免测试通过被当成安全证明。"""
    guard = detail.get("guard") if isinstance(detail.get("guard"), dict) else {}
    observed = category in ("correct", "numeric_mismatch", "metadata_mismatch")
    claim = {
        "correct": "此输入的输出与参考实现一致",
        "numeric_mismatch": "此输入观察到数值不一致",
        "metadata_mismatch": "此输入观察到形状或类型不一致",
    }.get(category, "此次运行未得到可分类的完整输出")
    return {
        "guard_basis": guard.get("evidence", "Unknown"),
        "guard_scope": guard.get("proof_scope", "未经过 Guard 静态判断"),
        "fast_eligible": bool(guard.get("allowed") and guard.get("evidence") == "Statically-Proven"),
        "observation_level": "Empirically-Validated" if observed else "Unknown",
        "observation_claim": claim,
        "observation_scope": "仅本次输入和运行环境；不证明其他输入安全" if observed else "无可外推的运行证据",
    }


def run_isolated(name: str, layout: str, *, m: int = 7, n: int = 11, mode: str = "dispatch", refined: bool = False, timeout: int = 90) -> OracleResult:
    root = Path(__file__).resolve().parent.parent
    command = [sys.executable, "-m", "scenarios.worker", name, layout, "--m", str(m), "--n", str(n), "--mode", mode]
    if refined:
        command.append("--refined")
    try:
        done = subprocess.run(command, cwd=root, capture_output=True, text=True, timeout=timeout, check=False)
    except subprocess.TimeoutExpired:
        detail = {"case": name, "layout": layout}
        return OracleResult("timeout_unknown", detail, None, "", _evidence("timeout_unknown", detail))
    lines = [line for line in done.stdout.splitlines() if line.strip()]
    try:
        detail = json.loads(lines[-1]) if lines else {}
    except json.JSONDecodeError:
        detail = {"stdout": done.stdout[-2000:]}
    if done.returncode == 0:
        category = detail.get("kind", "unknown")
    else:
        # CUDA Context 出错后信息可能不完整；无法确认的故障保留为未知。
        error_text = (detail.get("message", "") + done.stderr).lower()
        if "illegal memory access" in error_text or "out of bounds" in error_text:
            category = "memory_error_reported"
        elif "cuda" in error_text or "device-side assert" in error_text:
            category = "cuda_error_unclassified"
        elif done.returncode < 0:
            category = "process_failure_unknown"
        else:
            category = "exception_unknown"
    return OracleResult(category, detail, done.returncode, done.stderr[-2000:], _evidence(category, detail))
