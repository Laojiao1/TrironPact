import bench.e5.readiness as readiness
from bench.e5.readiness import build_ccfa, build_ccfb
from bench.e5.review import ReviewValidation


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


def test_second_review_requires_strict_record_and_all_current_risks(monkeypatch):
    risk = {"rows": [{"case_id": "r1", "status": "complete"}, {"case_id": "r2", "status": "complete"}]}
    monkeypatch.setattr(
        readiness,
        "validate_completed_review",
        lambda: ReviewValidation(True, (), ("k1",), ("r1",)),
    )
    assert readiness._second_review_complete(risk) is False
    monkeypatch.setattr(
        readiness,
        "validate_completed_review",
        lambda: ReviewValidation(True, (), ("k1",), ("r1", "r2")),
    )
    assert readiness._second_review_complete(risk) is True

