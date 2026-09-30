"""核对原生 Linux 候选资源与预备脚本；未取得机器时保持未通过。"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime
from pathlib import Path

from bench.e1.acquire import ROOT

PROJECT = ROOT.parent.parent
RESULTS = PROJECT / "results"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    data = json.loads((ROOT / "resources.json").read_text(encoding="utf-8"))
    setup = PROJECT / "scripts" / "e1" / "setup_native_linux.sh"
    smoke = PROJECT / "scripts" / "e1" / "native_smoke.py"
    candidates = data["candidates"]
    if len(candidates) != 2 or len({c["architecture"] for c in candidates}) != 2:
        raise ValueError("必须登记两个不同架构的候选资源")
    result = {
        **data,
        "generated_at": datetime.now().astimezone().isoformat(),
        "registry_sha256": digest(ROOT / "resources.json"),
        "preparation": {"setup_script": str(setup.relative_to(PROJECT)), "setup_sha256": digest(setup),
                        "smoke_script": str(smoke.relative_to(PROJECT)), "smoke_sha256": digest(smoke)},
        "development_environment": {"system": platform.platform(), "counts_as_native_linux_resource": False,
                                    "reason": "当前为 WSL2 开发环境，只用于实现和正确性 smoke。"},
        "checks": {"two_candidates_registered": True, "scripts_prepared": True,
                   "two_native_resources_acquired": all(c["access_status"] == "acquired" for c in candidates),
                   "two_native_smokes_passed": all(c["smoke_status"] == "passed" for c in candidates)},
    }
    lines = ["# e1 原生 Linux GPU 资源准备", "", f"- 登记日期：{data['as_of']}。", f"- 策略：{data['policy']}", "",
             "| 候选 | GPU/架构 | 系统与驱动 | 时段 | 成本 | 状态 |", "| --- | --- | --- | --- | --- | --- |"]
    for item in candidates:
        lines.append(f"| {item['id']} | {item['gpu']} / {item['architecture']} | {item['target_os']}；{item['driver']} | {item['availability_window']} | ${item['cost_usd_per_gpu_hour']}/GPU-hour | {item['access_status']}；smoke={item['smoke_status']} |")
    lines.extend(["", "候选、目标版本和环境/代表 Kernel smoke 脚本已冻结，满足 e1 的资源准备要求。根据 2026-09-30 用户决定，实际租赁与两机 smoke 延期到项目开发完成后、正式跨硬件实验前；当前 `native_resource_ready=false`，且未运行任何 e4 性能实验。"])
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "e1_native_linux_resources.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (RESULTS / "e1_native_linux_resources.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result["checks"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
