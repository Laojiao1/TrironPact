"""冻结 e2 开发集规则指纹，防止留出评估后继续调规则。"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

from bench.e2.coverage import PROJECT

FREEZE = PROJECT / "bench" / "e2" / "freeze.json"


def rule_paths() -> list[Path]:
    relative = (
        "pact/e2_access.py",
        "pact/e2_contracts.py",
        "pact/e2_guard.py",
        "bench/e2/coverage.py",
        "bench/e2/contracts.py",
        "bench/e2/plans.py",
        "bench/e2/worker.py",
        "bench/e2/guards.py",
        "test/test_e2_access_ir.py",
        "test/test_e2_contracts.py",
        "test/test_e2_guard.py",
        "bench/e1/catalog.json",
    )
    return [PROJECT / path for path in relative]


def current_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in rule_paths():
        digest.update(path.relative_to(PROJECT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def freeze() -> dict:
    data = {
        "frozen_at": datetime.now().astimezone().isoformat(),
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT, capture_output=True, text=True, check=True).stdout.strip(),
        "rule_fingerprint": current_fingerprint(),
        "paths": [path.relative_to(PROJECT).as_posix() for path in rule_paths()],
        "development_counts": {"supported": 21, "candidate_complete": 21, "guard_available": 21, "fast_feasible": 21},
        "holdout_evaluated": False,
        "policy": "冻结后仅执行一次留出评估；不得按留出结果调规则。规则变化必须作废该留出结果并建立新版本。",
    }
    FREEZE.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return data


def check() -> bool:
    if not FREEZE.exists():
        return False
    data = json.loads(FREEZE.read_text(encoding="utf-8"))
    return data.get("rule_fingerprint") == current_fingerprint()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    data = freeze() if args.freeze else json.loads(FREEZE.read_text(encoding="utf-8"))
    current = data.get("rule_fingerprint") == current_fingerprint()
    print(json.dumps({"rule_fingerprint_current": current, "holdout_evaluated": data.get("holdout_evaluated")}, ensure_ascii=False))
    return 0 if current else 2


if __name__ == "__main__":
    raise SystemExit(main())
