"""同预算的谓词边界与随机物理变异对照；所有原始执行均隔离。"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

from bench.metrics import select_witnesses
from pact.boundary_report import _case_candidates, _base_for
from pact.candidate_report import build_candidate_report
from pact.mutation import MutationRecipe, TensorRecipe, shadow_candidates, standard_recipes
from pact.mutation_oracle import run_mutation


ROOT = Path(__file__).resolve().parent.parent


def random_peer(recipe: MutationRecipe, seed: int) -> MutationRecipe:
    """同一 case/shape 下均匀抽取合法 stride/offset；不预先针对候选。"""
    rng = random.Random(seed)
    inputs = []
    for name, spec in recipe.inputs:
        if len(spec.shape) == 1:
            stride = (rng.choice((1, 2, 3, 4)),)
        else:
            n = spec.shape[1]
            stride = (rng.choice((n, n + 1, n + 3, 2 * n)), rng.choice((1, 2, 3)))
        offset = rng.randrange(0, 4)
        inputs.append((name, TensorRecipe(spec.shape, stride, offset)))
    return MutationRecipe(1, recipe.case, f"random_{seed}", tuple(inputs), recipe.target_family, seed)


def shadow_changed(recipe: MutationRecipe, candidates: list[dict]) -> tuple[int, bool]:
    before = shadow_candidates(_base_for(recipe), candidates)
    after = shadow_candidates(recipe, candidates)
    changed = sum(a["value"] is not None and b["value"] is not None and a["value"] != b["value"]
                  for a, b in zip(before, after) if a["evidence"] == "Statically-Proven")
    # 物理指针对齐仍需真实 Tensor；符号地址 0 不能提升为运行时放行。
    return changed, any(item["evidence"] == "Unproven" for item in after)


def _prepare(recipes: tuple[MutationRecipe, ...], report: dict) -> tuple[list[dict], dict]:
    rows = []
    for recipe in recipes:
        changed, dynamic_pointer = shadow_changed(recipe, _case_candidates(report, recipe.case))
        reason = ("decision_flip" if changed else "tile_boundary" if recipe.target_family == "tile_mask"
                  else "singleton" if "single" in recipe.label else "sample")
        rows.append({"id": recipe.id, "kernel": recipe.case, "reason": reason,
                     "changed_static_candidates": changed, "dynamic_pointer_unresolved": dynamic_pointer,
                     "recipe": recipe.to_dict()})
    return rows, {"generated": len(recipes), "unique": len({item["id"] for item in rows}),
                  "candidate_flips": sum(item["changed_static_candidates"] > 0 for item in rows)}


def run_arm(name: str, recipes: tuple[MutationRecipe, ...], report: dict, *, workers: int) -> dict:
    shadow, meta = _prepare(recipes, report)
    chosen = select_witnesses(shadow, max_per_kernel=max(1, workers), max_total=workers)["selected"]
    outcomes = []
    start = time.perf_counter()
    for item in chosen:
        recipe = MutationRecipe.from_dict(item["recipe"])
        before = time.perf_counter()
        result = run_mutation(recipe)
        outcomes.append({"id": recipe.id, "reason": item["reason"], "candidate_flips": item["changed_static_candidates"],
                         "category": result["category"], "elapsed_s": time.perf_counter() - before,
                         "oracle": result})
    counts = Counter(item["category"] for item in outcomes)
    return {"name": name, "shadow": meta, "worker_budget": workers, "workers_used": len(outcomes),
            "elapsed_s": time.perf_counter() - start, "categories": dict(counts),
            "unique_numeric_mismatches": len({item["id"] for item in outcomes if item["category"] == "numeric_mismatch"}),
            "review_overstrong_witnesses": len({item["id"] for item in outcomes if item["category"] == "correct" and item["candidate_flips"] > 0}),
            "review_understrong_conflicts": len({item["id"] for item in outcomes if item["category"] == "numeric_mismatch" and item["candidate_flips"] == 0}),
            "outcomes": outcomes, "selection": chosen,
            "conclusion_scope": "只比较这批有限配方在同一 worker 预算内的发现；不同生成分布及筛选会影响结果。"}


def build(*, workers: int = 8, seed: int = 20260925) -> dict:
    if not 1 <= workers <= 36:
        raise ValueError("隔离 worker 预算为 1～36")
    regression = json.loads((ROOT / "results/candidate_regression.json").read_text(encoding="utf-8"))
    report = build_candidate_report(regression)
    guided = standard_recipes()
    random_recipes = tuple(random_peer(recipe, seed + i) for i, recipe in enumerate(guided))
    arms = [run_arm("predicate_guided", guided, report, workers=workers),
            run_arm("uniform_stride_offset", random_recipes, report, workers=workers)]
    return {"generated_at": datetime.now().astimezone().isoformat(), "seed": seed, "arms": arms,
            "smt_real_deletions": [], "smt_fixture_only": True,
            "limits": "当前真实候选无可删除冗余对；边界结果为执行观察，不能单独削弱或放宽一般契约。"}


def render(data: dict) -> str:
    lines = ["# 第六阶段受限边界抽样对照", "", f"生成时间：{data['generated_at']}；种子 {data['seed']}。",
             "", "| 方法 | 生成/唯一 | 静态候选翻转 | 独立 worker | 数值错读 | 待审过强见证 | 待审过弱冲突 | 用时 (s) | 分类 |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for arm in data["arms"]:
        lines.append(f"| {arm['name']} | {arm['shadow']['generated']}/{arm['shadow']['unique']} | {arm['shadow']['candidate_flips']} | {arm['workers_used']} | {arm['unique_numeric_mismatches']} | {arm['review_overstrong_witnesses']} | {arm['review_understrong_conflicts']} | {arm['elapsed_s']:.1f} | {arm['categories']} |")
    lines += ["", "同 worker 预算不意味着两种生成分布拥有相同的有效边界样本；原始配方与逐例结果见 JSON。",
              "待审过强见证仅表示候选翻转但本次输出正确；待审过弱冲突仅表示无已知静态翻转但本次错读，均需逐条独立推导，不能自动精化。",
              "真实候选 SMT 删除：0；合成 fixture 不计入真实性能或精化收益。", data["limits"], ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "results/phase6_boundary_ablation.md")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260925)
    parser.add_argument("--rerender-existing", action="store_true", help="只从同名 JSON 重算派生计数，不再次启动 GPU worker")
    args = parser.parse_args()
    if args.rerender_existing:
        data = json.loads(args.output.with_suffix(".json").read_text(encoding="utf-8"))
        for arm in data["arms"]:
            outcomes = arm["outcomes"]
            arm["review_overstrong_witnesses"] = len({item["id"] for item in outcomes if item["category"] == "correct" and item["candidate_flips"] > 0})
            arm["review_understrong_conflicts"] = len({item["id"] for item in outcomes if item["category"] == "numeric_mismatch" and item["candidate_flips"] == 0})
    else:
        data = build(workers=args.workers, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(data), encoding="utf-8")
    args.output.with_suffix(".json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"guided={data['arms'][0]['workers_used']} random={data['arms'][1]['workers_used']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
