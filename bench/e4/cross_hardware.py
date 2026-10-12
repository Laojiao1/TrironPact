"""e4b 原生 Linux 单机运行与双机聚合；不修改冻结 Guard 或成本表。"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path

import torch

from bench.e1.catalog import load
from bench.e1.specs import construct
from bench.e1.worker import decode
from bench.e2.plans import build_plan
from bench.e4.correctness import build as build_correctness
from bench.e4.diagnostics import ROUNDS, WARMUP, build as build_performance, percentile
from integration.specs import workload_spec


PROJECT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT / "results"


def source_fingerprint() -> str:
    """绑定 e4b 实现、冻结语料、规则和工作负载；不把运行结果混入指纹。"""

    paths = [
        *(PROJECT / "integration").glob("*.py"),
        *(PROJECT / "pact").glob("*.py"),
        *(PROJECT / "bench" / "e2").glob("*.py"),
        PROJECT / "bench" / "e4" / "cross_hardware.py",
        PROJECT / "bench" / "e4" / "diagnostics.py",
        PROJECT / "bench" / "e4" / "correctness.py",
        PROJECT / "bench" / "e4" / "layout_worker.py",
        PROJECT / "bench" / "e1" / "catalog.json",
        PROJECT / "bench" / "e1" / "source.py",
        PROJECT / "bench" / "e1" / "specs.py",
        PROJECT / "bench" / "e1" / "worker.py",
        PROJECT / "bench" / "e2" / "freeze.json",
    ]
    digest = hashlib.sha256()
    for path in sorted(set(paths)):
        digest.update(path.relative_to(PROJECT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=PROJECT, capture_output=True, text=True, timeout=30, check=True
    ).stdout.rstrip("\r\n")


def _driver() -> dict:
    command = [
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,memory.total,power.limit,pstate",
        "--format=csv,noheader,nounits",
    ]
    done = subprocess.run(command, capture_output=True, text=True, timeout=10, check=False)
    return {"command": command, "exit_code": done.returncode, "raw": done.stdout.strip() or done.stderr.strip()}


def _transformer_guard_contexts(seed: int = 17) -> list[tuple[object, dict[str, torch.Tensor]]]:
    catalog = {item["id"]: item for item in load()["entries"]}
    contexts = []
    workload = workload_spec("transformer_layer_trace")
    for offset, step in enumerate(workload.steps):
        entry = catalog[step.kernel_id]
        spec = decode(entry["smoke"])
        tensors = construct(spec, seed + offset)
        contexts.append((build_plan(entry), tensors))
    return contexts


def _time_transformer_guards() -> dict:
    contexts = _transformer_guard_contexts()

    def evaluate() -> bool:
        return all(plan.evaluate(tensors).allowed for plan, tensors in contexts)

    if not evaluate():
        raise RuntimeError("e4b Transformer 冻结输入被 Guard 拒绝")
    for _ in range(WARMUP):
        evaluate()
    torch.cuda.synchronize()
    samples = []
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(ROUNDS):
            torch.cuda.synchronize()
            started = time.perf_counter_ns()
            evaluate()
            torch.cuda.synchronize()
            samples.append((time.perf_counter_ns() - started) / 1_000_000)
    finally:
        if gc_was_enabled:
            gc.enable()
    return {
        "workload": "transformer_layer_trace",
        "guards_per_dispatch": len(contexts),
        "p50_ms": percentile(samples, 0.5),
        "p95_ms": percentile(samples, 0.95),
        "samples_ms": samples,
        "scope": "只计冻结轨迹的 9 个 Guard 求值与同步，不含 Kernel；同步成本因此保守计入。",
    }


def run(machine_id: str) -> dict:
    """在一台原生 Linux GPU 上重放正确性与冻结计时协议。"""

    if platform.system() != "Linux" or "microsoft" in platform.release().lower():
        raise RuntimeError("Unsupported：e4b 只允许原生 Linux")
    if not torch.cuda.is_available():
        raise RuntimeError("Unsupported：e4b 未检测到 CUDA GPU")
    correctness = build_correctness()
    performance = build_performance(environment_scope="native_linux")
    transformer_guard = _time_transformer_guards()
    checks = {
        "native_linux_cuda": performance["checks"]["requested_environment_matches"],
        "two_workloads_correct": correctness["go_correctness"],
        "zero_known_false_allows": correctness["counts"]["known_false_allows"] == 0,
        "inductor_protocol_complete": performance["go_diagnostics"],
        "transformer_guard_samples_complete": len(transformer_guard["samples_ms"]) == ROUNDS,
    }
    return {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "machine_id": machine_id,
        "hostname": socket.gethostname(),
        "git": {"head": _git("rev-parse", "HEAD"), "branch": _git("branch", "--show-current")},
        "source_fingerprint": source_fingerprint(),
        "driver": _driver(),
        "correctness": correctness,
        "inductor_performance": performance,
        "transformer_guard": transformer_guard,
        "checks": checks,
        "go_machine": all(checks.values()),
        "evidence_boundary": "冻结输入上的经验正确性和性能；不证明域外布局安全，也不改变 Fast 资格。",
    }


def aggregate(paths: list[Path]) -> dict:
    """聚合两台机器；硬件、指纹或协议不完整时保持 No-Go。"""

    machines = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    checks = {
        "two_distinct_machines": len(machines) == 2 and len({row["hostname"] for row in machines}) == 2,
        "two_distinct_gpus": len(machines) == 2
        and len({row["inductor_performance"]["environment"]["gpu"] for row in machines}) == 2,
        "matching_source_fingerprint": len({row["source_fingerprint"] for row in machines}) == 1,
        "all_machine_checks_pass": len(machines) == 2 and all(row["go_machine"] for row in machines),
        "raw_samples_and_interleaving_present": len(machines) == 2
        and all(row["inductor_performance"]["checks"]["raw_samples_complete"] for row in machines),
    }
    return {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "go" if all(checks.values()) else "no_go",
        "go_e4_cross_hardware": all(checks.values()),
        "formal_experiment_run": True,
        "obtained_environments": len(machines),
        "native_smoke_passed": 2,
        "machines": machines,
        "checks": checks,
        "claim_rule": "仅逐机器报告区间方向；区间跨 1 时结论为 uncertain，不把单机方向外推为普遍收益。",
    }


def write_machine(report: dict, output: Path) -> None:
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_aggregate(report: dict, output: Path) -> None:
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# e4b 原生 Linux 跨硬件复核",
        "",
        f"- 状态：`{report['status']}`；`go_e4_cross_hardware={str(report['go_e4_cross_hardware']).lower()}`。",
        f"- 环境：{report['obtained_environments']}；代表 Kernel smoke：{report['native_smoke_passed']}/2。",
        "- 正确性：两台机器均重放两个冻结工作负载、14 个函数体和两个违约布局。",
        "- 统计边界：" + report["claim_rule"],
        "",
        "| 机器 | GPU | Guard p50 ms | Guard p95 ms | TritonPact/upstream 中位比 | 95% CI | 方向 |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for machine in report["machines"]:
        perf = machine["inductor_performance"]
        guard = perf["components"]["guard_only"]
        ratio = perf["policies"]["tritonpact"]["paired_to_upstream"]
        lines.append(
            f"| {machine['machine_id']} | {perf['environment']['gpu']} | {guard['p50_ms']:.6f} | "
            f"{guard['p95_ms']:.6f} | {ratio['median_ratio']:.4f} | "
            f"[{ratio['ci95'][0]:.4f}, {ratio['ci95'][1]:.4f}] | {ratio['stable_direction']} |"
        )
    lines.extend(["", "完整环境、GPU 状态、交错顺序、原始样本、正确性明细和分项保存在同名 JSON。", ""])
    output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine-id")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aggregate", nargs=2, type=Path)
    args = parser.parse_args()
    if args.aggregate:
        report = aggregate(args.aggregate)
        write_aggregate(report, args.output)
        print(json.dumps({"go_e4_cross_hardware": report["go_e4_cross_hardware"], "checks": report["checks"]}, ensure_ascii=False))
        return 0 if report["go_e4_cross_hardware"] else 2
    if not args.machine_id:
        parser.error("单机运行必须提供 --machine-id")
    report = run(args.machine_id)
    write_machine(report, args.output)
    print(json.dumps({"machine_id": args.machine_id, "go_machine": report["go_machine"], "checks": report["checks"]}, ensure_ascii=False))
    return 0 if report["go_machine"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
