"""冻结最近邻工作与一个可执行外部基线；不修改 TritonPact 在线路径。"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
import venv
from datetime import datetime
from pathlib import Path

from bench.e1.acquire import ROOT

PROJECT = ROOT.parent.parent
RESULTS = PROJECT / "results"
SOURCES = ROOT / "related_sources.json"


def _run(command: list[str], cwd: Path, timeout: int) -> dict:
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False)
    return {
        "command": command,
        "exit_code": completed.returncode,
        "stdout": completed.stdout[-8000:],
        "stderr": completed.stderr[-8000:],
    }


def run_triton_verify() -> dict:
    """在临时 venv 复现固定提交的 safe/bug 示例；失败显式返回 Unknown。"""
    source = next(item for item in json.loads(SOURCES.read_text(encoding="utf-8"))["works"] if item["id"] == "triton_verify")
    revision = source["revision"]
    archive_url = f"https://codeload.github.com/{source['repository']}/tar.gz/{revision}"
    try:
        request = urllib.request.Request(archive_url, headers={"User-Agent": "TritonPact-e1-baseline-audit"})
        with urllib.request.urlopen(request, timeout=60) as response:
            archive = response.read()
    except urllib.error.URLError as error:
        return {"status": "Unknown", "reason": "archive_download_failed", "detail": str(error), "archive_url": archive_url}

    with tempfile.TemporaryDirectory(prefix="tritonpact-e1-baseline-") as raw_tmp:
        temp = Path(raw_tmp)
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
            for member in bundle.getmembers():
                relative = Path(member.name)
                if relative.is_absolute() or ".." in relative.parts or member.issym() or member.islnk():
                    return {"status": "Unknown", "reason": "unsafe_archive_member", "member": member.name}
            bundle.extractall(temp)
        checkout = temp / f"triton-verify-{revision}"
        environment = temp / "venv"
        venv.EnvBuilder(with_pip=True).create(environment)
        python = environment / "bin" / "python"
        install = _run([str(python), "-m", "pip", "install", "--disable-pip-version-check", "lark==1.2.2", "z3-solver==4.15.3.0"], checkout, 240)
        if install["exit_code"] != 0:
            return {"status": "Unknown", "reason": "dependency_install_failed", "archive_url": archive_url,
                    "archive_sha256": hashlib.sha256(archive).hexdigest(), "install": install}
        safe = _run([str(python), "main.py", "real_vector_add.ttir", "512", "4"], checkout, 60)
        bug = _run([str(python), "main.py", "real_vector_add.ttir", "500", "4"], checkout, 60)
        passed = safe["exit_code"] == 0 and "SAFE" in safe["stdout"] and bug["exit_code"] == 0 and "BUG FOUND" in bug["stdout"]
        return {
            "status": "passed" if passed else "failed",
            "archive_url": archive_url,
            "archive_sha256": hashlib.sha256(archive).hexdigest(),
            "revision": revision,
            "dependencies": {"lark": "1.2.2", "z3-solver": "4.15.3.0"},
            "install": install,
            "cases": {"safe_512": safe, "bug_500": bug},
            "scope": "仅复现公开 TTIR 指针边界示例；不代表 TritonPact 等价、完整或更安全。",
        }


def validate_sources(data: dict) -> None:
    """公开代码必须固定完整提交；无代码工作必须显式登记为定性比较。"""
    if data.get("as_of") != "2026-09-30" or not data.get("primary_venue_direction"):
        raise ValueError("相关工作审计日期或投稿方向未冻结")
    for work in data["works"]:
        if work["repository"] is not None and (not work["revision"] or len(work["revision"]) != 40):
            raise ValueError(f"公开代码未固定完整提交：{work['id']}")
        if work["repository"] is None and work["baseline"] != "qualitative_only":
            raise ValueError(f"无公开代码的工作只能定性比较：{work['id']}")


def write_report(baseline: dict | None) -> dict:
    data = json.loads(SOURCES.read_text(encoding="utf-8"))
    validate_sources(data)
    result = {
        **data,
        "generated_at": datetime.now().astimezone().isoformat(),
        "fingerprints": {"registry_sha256": hashlib.sha256(SOURCES.read_bytes()).hexdigest(),
                         "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "executable_baseline": baseline,
        "checks": {
            "first_party_sources_reviewed": len(data["works"]) >= 5,
            "public_code_pinned": all(w["repository"] is None or len(w["revision"]) == 40 for w in data["works"]),
            "primary_direction_frozen": bool(data["primary_venue_direction"]),
            "external_baseline_reproduced": bool(baseline and baseline.get("status") == "passed"),
        },
    }
    lines = [
        "# e1 最近邻工作与基线冻结",
        "",
        f"- 核对日期：{data['as_of']}。",
        f"- 主要投稿方向：{data['primary_venue_direction']['name']}。",
        f"- 外部可执行基线：triton-verify，状态 `{baseline.get('status') if baseline else 'not_run'}`。",
        "",
        "| 工作 | 正式状态 | 代码/基线 | 与 TritonPact 的边界 |",
        "| --- | --- | --- | --- |",
    ]
    for work in data["works"]:
        lines.append(f"| {work['title']} | {work['publication']} | {work['baseline']} | {work['boundary']} |")
    lines.extend(["", "详细固定提交、复现命令、stdout/stderr 和证据边界见同名 JSON。无法运行的工作只作定性比较。"])
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "e1_related_work_baselines.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (RESULTS / "e1_related_work_baselines.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-triton-verify", action="store_true")
    args = parser.parse_args()
    existing_path = RESULTS / "e1_related_work_baselines.json"
    existing = json.loads(existing_path.read_text(encoding="utf-8")) if existing_path.exists() else {}
    baseline = run_triton_verify() if args.run_triton_verify else existing.get("executable_baseline")
    result = write_report(baseline)
    print(json.dumps(result["checks"], ensure_ascii=False))
    return 0 if all(result["checks"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
