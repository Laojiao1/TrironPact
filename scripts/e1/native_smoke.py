"""原生 Linux 候选机只读 smoke；输出环境指纹，不产生性能结论。"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch
import triton


def main() -> int:
    proc_version = Path("/proc/version").read_text(encoding="utf-8", errors="replace")
    if sys.platform != "linux" or "microsoft" in proc_version.lower():
        print(json.dumps({"status": "Unsupported", "reason": "native_linux_required"}, ensure_ascii=False))
        return 2
    if not torch.cuda.is_available():
        print(json.dumps({"status": "Unknown", "reason": "cuda_unavailable"}, ensure_ascii=False))
        return 2
    driver = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                            capture_output=True, text=True, timeout=20, check=False)
    if driver.returncode != 0:
        print(json.dumps({"status": "Unknown", "reason": "nvidia_smi_failed", "stderr": driver.stderr[-2000:]}, ensure_ascii=False))
        return 2
    from bench.e1.worker import execute

    kernel = execute("triton_add")
    passed = kernel["status"] == "passed"
    result = {
        "status": "passed" if passed else "failed",
        "generated_at": datetime.now().astimezone().isoformat(),
        "system": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "triton": triton.__version__,
        "cuda": torch.version.cuda,
        "gpu_driver": driver.stdout.strip(),
        "representative_kernel": kernel,
        "scope": "只读环境与正确性 smoke；没有 warmup、计时或性能结论。",
    }
    print(json.dumps(result, ensure_ascii=False))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
