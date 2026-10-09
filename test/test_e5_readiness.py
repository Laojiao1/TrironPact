import json

import bench.e5.readiness as readiness
from bench.e5.readiness import build_ccfa, build_ccfb


def test_ccfb_readiness_fails_closed_on_external_and_final_freeze_gaps():
    report = build_ccfb()
    assert report["go_e5_ccfb_readiness"] is False
    assert report["go_e5"] is False
    failed = {row["id"] for row in report["missing"]}
    assert failed == {
        "4_two_workloads_and_two_hardware_environments",
        "7_independent_second_review",
        "9_current_commit_environment_and_data_bound",
        "10_clean_checkout_one_click_verification",
    }


def test_ccfa_gap_has_no_quantity_based_go():
    report = build_ccfa()
    assert report["go_e5_ccfa"] is False
    assert len(report["gaps"]) >= 5
    assert all(row["status"] == "open" for row in report["gaps"])


def test_second_review_requires_independence_quarter_sample_and_all_risks(tmp_path, monkeypatch):
    monkeypatch.setattr(readiness, "RESULTS", tmp_path)
    risk = {"rows": [{"case_id": "r1", "status": "complete"}, {"case_id": "r2", "status": "complete"}]}
    incomplete = {
        "reviewer_independent": True,
        "reviewed_kernel_count": 7,
        "formal_kernel_denominator": 28,
        "reviewed_risk_case_ids": ["r1"],
        "disagreements_resolved": True,
        "review_records": [{"kernel_id": "k1"}],
    }
    (tmp_path / "e5_second_review.json").write_text(json.dumps(incomplete), encoding="utf-8")
    assert readiness._second_review_complete(risk) is False
    incomplete["reviewed_risk_case_ids"] = ["r1", "r2"]
    (tmp_path / "e5_second_review.json").write_text(json.dumps(incomplete), encoding="utf-8")
    assert readiness._second_review_complete(risk) is True

