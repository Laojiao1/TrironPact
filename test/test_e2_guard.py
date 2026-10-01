"""e2 GuardPlan 的编译完整性和 fail-closed 行为。"""

from dataclasses import replace

import pytest
import torch

from bench.e1.catalog import load
from bench.e1.worker import decode
from bench.e1.specs import construct
from bench.e2.plans import build_plan


def _entry(kernel_id: str) -> dict:
    return next(item for item in load()["entries"] if item["id"] == kernel_id)


def test_development_plans_compile_without_kernel_name_rules():
    entries = [item for item in load()["entries"] if item["split"] == "development" and item["smoke"]]
    plans = [build_plan(item) for item in entries]
    assert len(plans) == 21
    assert all(plan.status == "Supported" for plan in plans)
    assert all(len(plan.fingerprint) == 64 for plan in plans)
    assert all({candidate.purpose for candidate in plan.candidates} == {"Safety", "Semantics"} for plan in plans)


def test_incomplete_candidate_plan_is_unknown():
    plan = build_plan(_entry("triton_add"))
    broken = replace(plan, candidates=())
    assert broken.evaluate({}).status == "Unknown"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_guard_accepts_frozen_domain_and_rejects_layout_alias_and_dtype():
    entry = _entry("triton_add")
    plan = build_plan(entry)
    tensors = construct(decode(entry["smoke"]), 17)
    assert plan.evaluate(tensors).allowed
    transposed = {**tensors, "x": torch.empty((1, 129), device="cuda").T}
    assert not plan.evaluate(transposed).allowed
    aliased = {**tensors, "out": tensors["x"]}
    assert plan.evaluate(aliased).status == "Unknown"
    wrong_dtype = {**tensors, "x": tensors["x"].half()}
    assert plan.evaluate(wrong_dtype).status == "False"
