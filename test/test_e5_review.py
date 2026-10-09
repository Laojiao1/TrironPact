import copy
import json
from pathlib import Path

from bench.e5.review import MANIFEST, TEMPLATE, validate_manifest, validate_review


NOW = "2026-10-09T10:00:00+08:00"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _completed() -> tuple[dict, dict]:
    manifest = _load(MANIFEST)
    data = copy.deepcopy(_load(TEMPLATE))
    data["status"] = "complete"
    data["reviewer"] = {
        "reviewer_id": "reviewer-001",
        "independent": True,
        "no_prior_rule_or_label_involvement": True,
        "independence_declaration": "我未参与规则设计、语义标注或预期标签编写，并先独立完成初始判断。",
        "started_at": NOW,
        "completed_at": NOW,
    }
    data["reviewed_kernel_count"] = 7
    data["reviewed_risk_case_ids"] = [row["case_id"] for row in manifest["risk_cases"]]
    data["disagreements_resolved"] = True
    for record in data["review_records"]:
        record["initial_assessment"] = {
            "semantic_correspondence": "confirmed",
            "wrapper_binding": "confirmed",
            "reference_truth": "confirmed",
            "access_label": "Supported",
            "evidence": ["source:path#line", "reference:path#anchor"],
            "notes": "independent fixture",
            "completed_at": NOW,
            "frozen_before_reconciliation": True,
        }
        record["reconciliation"] = {
            "project_label_revealed_after_initial": True,
            "agrees_with_project": True,
            "final_status": "confirmed",
            "resolution_status": "not_needed",
            "resolution_evidence": [],
        }
    for record in data["risk_review_records"]:
        record["initial_assessment"] = {
            "observed_behavior": "numeric_mismatch",
            "reference_truth": "confirmed",
            "guard_behavior": "blocked",
            "risk_levels": ["L1"],
            "upstream_vulnerability_claim": "not_claimed",
            "evidence": ["worker:stdout-sha256", "reference:checks"],
            "notes": "independent fixture",
            "completed_at": NOW,
            "frozen_before_reconciliation": True,
        }
        record["reconciliation"] = {
            "project_label_revealed_after_initial": True,
            "agrees_with_project": True,
            "final_status": "confirmed",
            "resolution_status": "not_needed",
            "resolution_evidence": [],
        }
    return data, manifest


def test_review_manifest_is_bound_to_frozen_git_blobs_and_stratified():
    manifest = _load(MANIFEST)
    assert validate_manifest(manifest) == ()
    assert manifest["freeze_commit"] == "bd5fa3e3376e2dacffc2671ee52f3cbea1589a10"
    assert len(manifest["kernels"]) == 7
    assert {row["source"] for row in manifest["kernels"]} == {"triton", "pytorch", "liger", "flag_gems", "unsloth"}
    assert {row["family"] for row in manifest["kernels"]} == {"elementwise_mapping", "feature_broadcast", "layout_2d", "row_reduction"}
    assert {row["split"] for row in manifest["kernels"]} == {"development", "holdout"}
    assert len(manifest["risk_cases"]) == 4
    assert all("classification" not in row and "risk_levels" not in row for row in manifest["risk_cases"])


def test_unfilled_template_fails_closed():
    result = validate_review(_load(TEMPLATE), _load(MANIFEST))
    assert result.complete is False
    assert result.errors


def test_complete_independent_review_fixture_passes():
    data, manifest = _completed()
    result = validate_review(data, manifest)
    assert result.complete, result.errors
    assert len(result.kernel_ids) == 7
    assert len(result.risk_ids) == 4


def test_duplicate_kernel_and_unrecorded_disagreement_are_rejected():
    data, manifest = _completed()
    data["review_records"][1]["kernel_id"] = data["review_records"][0]["kernel_id"]
    data["risk_review_records"][0]["reconciliation"].update(
        {"agrees_with_project": False, "resolution_status": "resolved", "resolution_evidence": ["resolution:note"]}
    )
    result = validate_review(data, manifest)
    assert result.complete is False
    assert any("重复 kernel_id" in error for error in result.errors)
    assert any("disagreements" in error for error in result.errors)


def test_manifest_tampering_is_rejected():
    data, manifest = _completed()
    manifest["kernels"][0]["source_file_sha256"] = "0" * 64
    result = validate_review(data, manifest)
    assert result.complete is False
    assert any("冻结来源或哈希失配" in error for error in result.errors)
