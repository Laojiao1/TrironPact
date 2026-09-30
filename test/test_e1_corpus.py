"""e1 审计拒绝边界；不增加历史在线 Fast 的支持范围。"""

import json

import pytest

from bench.e1.catalog import CATALOG, load
from bench.e1.related import validate_sources
from bench.e1.source import body_fingerprint
from bench.e1.specs import value


def test_renaming_and_configuration_do_not_create_new_body():
    one = "@triton.jit\ndef add(X, Y):\n idx = tl.arange(0, 128)\n tl.store(Y + idx, tl.load(X + idx))"
    two = "@triton.jit(debug=False)\ndef renamed(a, b):\n i = tl.arange(0, 128)\n tl.store(b + i, tl.load(a + i))"
    assert body_fingerprint(one) == body_fingerprint(two)
    assert body_fingerprint(one) != body_fingerprint(two.replace("tl.load(a + i)", "-tl.load(a + i)"))


def test_catalog_preserves_challenge_denominator_and_split():
    data = load()
    assert len(data["entries"]) == 40
    positives = [r for r in data["entries"] if r["smoke"]]
    challenges = [r for r in data["entries"] if r["split"] == "challenge"]
    assert len(positives) == 28
    assert len(challenges) == 12
    assert len({r["normalized_body_sha256"] for r in data["entries"]}) == 40
    assert len({r["source"] for r in data["entries"]}) >= 4
    assert all(r["repository"] and len(r["revision"]) == 40 for r in data["entries"])
    assert all(r["smoke"] is None and not r["test_file"] and not r["reference_anchor"] for r in challenges)
    assert {r["split"] for r in data["entries"]} == {"development", "holdout", "challenge"}
    assert all(r["fast_status"] == "not_eligible" for r in data["entries"])


def test_positive_families_sources_and_bindings_reach_frozen_thresholds():
    data = load()
    positives = [r for r in data["entries"] if r["smoke"]]
    families = {name: [r for r in positives if r["family"] == name] for name in
                ("elementwise_mapping", "layout_2d", "feature_broadcast", "row_reduction")}
    assert all(len(rows) >= 5 for rows in families.values())
    assert len({r["source"] for r in positives}) == 5
    assert all(len(r["smoke"]["bindings"]) == len(r["signature"]) for r in positives)
    assert all(r["reference_anchor"] and r["reference_file_sha256"] for r in positives)


def test_related_work_and_native_resource_registries_are_explicit():
    related = json.loads((CATALOG.parent / "related_sources.json").read_text(encoding="utf-8"))
    validate_sources(related)
    resources = json.loads((CATALOG.parent / "resources.json").read_text(encoding="utf-8"))
    assert resources["execution_gate"] == "deferred_by_user_until_project_development_complete"
    assert len(resources["candidates"]) == 2
    assert len({row["architecture"] for row in resources["candidates"]}) == 2
    assert all(row["access_status"] == "not_acquired" and row["smoke_status"] == "not_run"
               for row in resources["candidates"])


@pytest.mark.parametrize("mutation", ["source", "binding", "split"])
def test_tampered_provenance_binding_or_split_rejected(tmp_path, mutation):
    data = json.loads(CATALOG.read_text(encoding="utf-8"))
    row = data["entries"][0]
    if mutation == "source": row["file_sha256"] = "0" * 64
    if mutation == "binding": row["smoke"]["bindings"].pop()
    if mutation == "split": row["split"] = "challenge"
    path = tmp_path / "catalog.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        load(path)


def test_unknown_wrapper_expression_cannot_execute_code():
    with pytest.raises(ValueError):
        value("__import__('os').system('bad')", {})
