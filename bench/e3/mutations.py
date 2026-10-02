"""在冻结共享域和不可变 GPU Oracle 上比较四种边界生成方法。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from pact.e3_mutation import Method, ordered_probes, probe_domain


PROJECT = Path(__file__).resolve().parents[2]
RESULTS = PROJECT / "results"
METHODS: tuple[Method, ...] = ("static_candidate", "uniform_random", "constrained_random", "predicate_guided")
SEEDS = (17, 42, 73, 101, 211)
ATTEMPT_BUDGET = len(probe_domain())
WORKER_BUDGET = 4
WALL_CLOCK_BUDGET_MS = 120_000.0


def _oracle() -> dict[str, dict]:
    report = json.loads((RESULTS / "e3_real_risk_cases.json").read_text(encoding="utf-8"))
    result = {}
    for row in report["rows"]:
        if row["status"] == "complete":
            result[row["case_id"]] = row["detail"]
    return result


def _trial(method: Method, seed: int, oracle: dict[str, dict]) -> dict:
    attempts = 0
    worker_calls = 0
    elapsed = 0.0
    witnesses = []
    first = None
    classifications = Counter()
    selected = []
    for probe in ordered_probes(method, seed):
        if attempts >= ATTEMPT_BUDGET or worker_calls >= WORKER_BUDGET or elapsed >= WALL_CLOCK_BUDGET_MS:
            break
        attempts += 1
        row = {
            "probe": probe.id,
            "fingerprint": probe.fingerprint,
            "family": probe.family,
            "predicate_flips": probe.predicate_flips,
            "worker": False,
        }
        if not probe.executable:
            row.update({"classification": "unreachable", "reason": probe.unreachable_reason})
            classifications["unreachable"] += 1
        elif probe.empirical_case is None:
            category = "mixed_unattributed" if len(probe.changed_inputs) > 1 else "metadata_boundary"
            row["classification"] = category
            classifications[category] += 1
        elif probe.empirical_case not in oracle:
            row["classification"] = "Unknown"
            classifications["Unknown"] += 1
        else:
            detail = oracle[probe.empirical_case]
            worker_calls += 1
            elapsed += float(detail["worker_elapsed_ms"])
            category = detail["classification"]
            row.update({"worker": True, "case_id": probe.empirical_case, "classification": category})
            classifications[category] += 1
            if category == "numeric_mismatch":
                witnesses.append(probe.empirical_case)
                if first is None:
                    first = {"attempt": attempts, "worker_call": worker_calls, "elapsed_ms": elapsed, "case_id": probe.empirical_case}
        selected.append(row)
    return {
        "method": method,
        "seed": seed,
        "attempt_budget": ATTEMPT_BUDGET,
        "worker_budget": WORKER_BUDGET,
        "wall_clock_budget_ms": WALL_CLOCK_BUDGET_MS,
        "attempts_used": attempts,
        "worker_calls": worker_calls,
        "worker_elapsed_ms": elapsed,
        "first_witness": first,
        "unique_witnesses": sorted(set(witnesses)),
        "classifications": dict(classifications),
        "selected": selected,
    }


def build() -> dict:
    oracle = _oracle()
    trials = [_trial(method, seed, oracle) for method in METHODS for seed in SEEDS]
    summary = {}
    for method in METHODS:
        rows = [row for row in trials if row["method"] == method]
        attempts = [row["first_witness"]["attempt"] for row in rows if row["first_witness"]]
        times = [row["first_witness"]["elapsed_ms"] for row in rows if row["first_witness"]]
        summary[method] = {
            "trials": len(rows),
            "first_witness_successes": len(attempts),
            "median_attempts_to_first": sorted(attempts)[len(attempts) // 2] if attempts else None,
            "median_worker_ms_to_first": sorted(times)[len(times) // 2] if times else None,
            "unique_witnesses": sorted({item for row in rows for item in row["unique_witnesses"]}),
        }
    guided = summary["predicate_guided"]
    baselines = [summary[name] for name in METHODS if name != "predicate_guided"]
    guided_advantage = bool(
        guided["median_attempts_to_first"] is not None
        and sum(guided["median_attempts_to_first"] < row["median_attempts_to_first"] for row in baselines if row["median_attempts_to_first"] is not None) >= 2
    )
    families = {probe.family for probe in probe_domain()}
    checks = {
        "same_domain_and_budgets": all(
            row["attempt_budget"] == ATTEMPT_BUDGET
            and row["worker_budget"] == WORKER_BUDGET
            and row["wall_clock_budget_ms"] == WALL_CLOCK_BUDGET_MS
            for row in trials
        ),
        "same_seed_pool": all({row["seed"] for row in trials if row["method"] == method} == set(SEEDS) for method in METHODS),
        "all_condition_families_covered": families == {
            "multi_axis_grid", "stride_broadcast_2d", "separate_inputs", "offset_alignment", "span_boundary", "reduction_row_length"
        },
        "unreachable_reasons_complete": all(probe.unreachable_reason for probe in probe_domain() if not probe.executable),
        "oracle_identity_complete": set(oracle) == {probe.empirical_case for probe in probe_domain() if probe.empirical_case},
        "every_method_finds_witness": all(summary[method]["first_witness_successes"] == len(SEEDS) for method in METHODS),
    }
    return {
        "generated_at": datetime.now().astimezone().isoformat(),
        "protocol": {
            "methods": METHODS,
            "seeds": SEEDS,
            "attempt_budget": ATTEMPT_BUDGET,
            "worker_budget": WORKER_BUDGET,
            "wall_clock_budget_ms": WALL_CLOCK_BUDGET_MS,
            "oracle": "e3_real_risk_cases.json；每个唯一案例仅独立执行一次，比较阶段作不可变成本重放",
        },
        "domain": [probe.__dict__ | {"fingerprint": probe.fingerprint} for probe in probe_domain()],
        "trials": trials,
        "summary": summary,
        "guided_advantage": guided_advantage,
        "claim": "谓词引导在多数基线的首次见证尝试数上更少" if guided_advantage else "谓词引导未在多数基线上占优，降为可解释边界生成方法",
        "checks": checks,
        "go_mutation": all(checks.values()),
    }


def write(report: dict) -> None:
    base = RESULTS / "e3_mutation_comparison"
    base.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# e3 变异方法公平对照",
        "",
        f"- 生成时间：{report['generated_at']}。",
        f"- 状态：`{'go' if report['go_mutation'] else 'no_go'}`。",
        f"- 冻结协议：`{report['protocol']}`。",
        f"- 贡献定位：{report['claim']}。",
        "",
        "| 方法 | 试验 | 首次见证成功 | 中位尝试数 | 中位 worker ms | 唯一见证 |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for method, row in report["summary"].items():
        lines.append(
            f"| {method} | {row['trials']} | {row['first_witness_successes']} | {row['median_attempts_to_first']} | "
            f"{row['median_worker_ms_to_first']} | {row['unique_witnesses']} |"
        )
    lines.extend(
        [
            "",
            "- 元数据快筛不计作经验风险见证；只有独立 GPU worker 的 `numeric_mismatch` 进入唯一见证数。",
            "- 混合变异单列为 `mixed_unattributed`，不作单谓词因果归因；非法短 storage 只保留不可达理由，不送 GPU。",
            "- `worker_elapsed_ms` 是不可变 Oracle 的成本重放，用于公平截断，不是 e4 端到端性能测量。",
            "",
        ]
    )
    base.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    report = build()
    write(report)
    print(json.dumps({"go_mutation": report["go_mutation"], "guided_advantage": report["guided_advantage"], "summary": report["summary"]}, ensure_ascii=False))
    return 0 if report["go_mutation"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

