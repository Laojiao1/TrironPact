"""第六阶段统计口径：独立标签与有限执行观察分开。"""

from __future__ import annotations

import math
import random
import statistics
from collections import Counter, defaultdict


TRUTHS = {"eligible", "ineligible", "unlabeled"}
DECISIONS = {"True", "False", "Unknown", "Unsupported"}


def decision_metrics(rows: list[dict]) -> dict:
    """只在独立标签且判定明确的输入上计算混淆矩阵。"""
    counts = Counter()
    groups: dict[str, list[dict]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        identity, truth, decision = row["id"], row["truth"], row["decision"]
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError("样例 ID 缺失或重复")
        if truth not in TRUTHS or decision not in DECISIONS:
            raise ValueError("标签或判定无效")
        if truth != "unlabeled" and not row.get("truth_source"):
            raise ValueError("独立真值缺少审计来源")
        seen.add(identity)
        groups[row.get("kernel", "unassigned")].append(row)
        counts["total"] += 1
        if truth == "unlabeled":
            counts["unlabeled"] += 1
            continue
        counts["labeled"] += 1
        if decision in {"Unknown", "Unsupported"}:
            counts[decision] += 1
            continue
        counts["decided"] += 1
        label = ("tp" if truth == "eligible" else "fp") if decision == "True" else ("fn" if truth == "eligible" else "tn")
        counts[label] += 1
    tp, fp, tn, fn = (counts[key] for key in ("tp", "fp", "tn", "fn"))
    ratio = lambda numerator, denominator: numerator / denominator if denominator else None
    by_kernel = {}
    for name, values in sorted(groups.items()):
        group = Counter()
        for row in values:
            if row["truth"] == "unlabeled":
                continue
            group["labeled"] += 1
            if row["decision"] in {"Unknown", "Unsupported"}:
                group["unknown"] += 1
                continue
            key = ("tp" if row["truth"] == "eligible" else "fp") if row["decision"] == "True" else ("fn" if row["truth"] == "eligible" else "tn")
            group[key] += 1
        by_kernel[name] = {**{key: group[key] for key in ("labeled", "unknown", "tp", "fp", "tn", "fn")},
                           "precision": ratio(group["tp"], group["tp"] + group["fp"]),
                           "recall_on_decided": ratio(group["tp"], group["tp"] + group["fn"])}
    macro = {}
    for key in ("precision", "recall_on_decided"):
        defined = [row[key] for row in by_kernel.values() if row[key] is not None]
        macro[key] = {"value": statistics.mean(defined) if defined else None, "defined_kernels": len(defined)}
    return {"counts": {key: counts[key] for key in ("total", "unlabeled", "labeled", "decided", "tp", "fp", "tn", "fn", "Unknown", "Unsupported")},
            "precision": ratio(tp, tp + fp), "recall_on_decided": ratio(tp, tp + fn),
            "false_rejection_on_decided": ratio(fn, tp + fn),
            "decision_coverage": ratio(counts["decided"], counts["labeled"]),
            "unknown_share_labeled": ratio(counts["Unknown"] + counts["Unsupported"], counts["labeled"]),
            "by_kernel": by_kernel, "macro_by_kernel": macro,
            "meaning": "Precision/recall 仅作用于有独立标签且已明确判定的输入；Unknown/Unsupported 单列，recall 不能解释为全清单召回。"}


def paired_speedups(rows: list[dict], baseline: str, candidate: str) -> dict:
    """只汇总同一输入上的已核对安全路径；其余显式排除。"""
    ratios: list[float] = []
    excluded = []
    for row in rows:
        paths = row.get("paths", {})
        if not row.get("same_semantics") or any(path not in paths or not paths[path].get("safe") for path in (baseline, candidate)):
            excluded.append({"id": row.get("id"), "reason": "语义或安全资格未核对"})
            continue
        base = paths[baseline].get("median_us")
        target = paths[candidate].get("median_us")
        if any(type(value) not in (int, float) or not math.isfinite(value) or value <= 0 for value in (base, target)):
            excluded.append({"id": row.get("id"), "reason": "同步延迟缺失或无效"})
            continue
        ratios.append(base / target)
    return {"paired_count": len(ratios), "speedups": ratios,
            "geomean": math.exp(statistics.mean(math.log(value) for value in ratios)) if ratios else None,
            "median": statistics.median(ratios) if ratios else None, "excluded": excluded}


def paired_round_interval(rows: list[dict], baseline: str, candidate: str, *, seed: int = 17, draws: int = 500) -> dict:
    """只对已测同一组输入和轮次做条件 bootstrap；不外推未选 Kernel。"""
    if draws < 100:
        raise ValueError("bootstrap 次数不足")
    matched = []
    for row in rows:
        paths = row.get("paths", {})
        if not row.get("same_semantics") or any(key not in paths or not paths[key].get("safe") for key in (baseline, candidate)):
            continue
        a, b = paths[baseline].get("samples_us"), paths[candidate].get("samples_us")
        if not a or not b or len(a) != len(b) or any(not math.isfinite(x) or x <= 0 for x in a + b):
            continue
        matched.append([math.log(x / y) for x, y in zip(a, b)])
    if not matched:
        return {"paired_cases": 0, "point_geomean": None, "ci95": None}
    rng = random.Random(seed)
    point = math.exp(statistics.mean(statistics.median(values) for values in matched))
    draws_log = []
    for _ in range(draws):
        draws_log.append(statistics.mean(statistics.median(rng.choices(values, k=len(values))) for values in matched))
    draws_log.sort()
    return {"paired_cases": len(matched), "point_geomean": point,
            "ci95": (math.exp(draws_log[int(0.025 * draws)]), math.exp(draws_log[min(draws - 1, int(0.975 * draws))])),
            "seed": seed, "draws": draws,
            "scope": "仅对同批输入的轮次重采样；不包括选型、硬件或环境变化的不确定性。"}


def select_witnesses(rows: list[dict], *, max_per_kernel: int, max_total: int) -> dict:
    """元数据层按优先级选见证；不赋予 Fast 资格。"""
    if max_per_kernel < 1 or max_total < 1:
        raise ValueError("worker 预算必须为正")
    priority = {"decision_flip": 0, "first_violation": 1, "singleton": 2, "tile_boundary": 3, "sample": 4}
    seen: set[str] = set()
    selected = []
    counts = Counter()
    for row in sorted(rows, key=lambda item: (priority.get(item["reason"], 5), item["kernel"], item["id"])):
        if row["id"] in seen:
            continue
        seen.add(row["id"])
        if counts[row["kernel"]] >= max_per_kernel or len(selected) >= max_total:
            continue
        selected.append(row)
        counts[row["kernel"]] += 1
    return {"generated": len(rows), "unique": len(seen), "selected": selected,
            "worker_budget": max_total, "per_kernel_budget": max_per_kernel,
            "note": "仅挑选待物理核对样例；纯元数据判定不等于完整 Guard 或 GPU 执行证据。"}
