from bench.e5.readiness import build_ccfa, build_ccfb


def test_ccfb_readiness_fails_closed_on_external_and_final_freeze_gaps():
    report = build_ccfb()
    assert report["go_e5_ccfb_readiness"] is False
    assert report["go_e5"] is False
    failed = {row["id"] for row in report["missing"]}
    assert failed == {
        "8_current_commit_environment_and_data_bound",
        "9_clean_checkout_one_click_verification",
    }


def test_ccfa_gap_has_no_quantity_based_go():
    report = build_ccfa()
    assert report["go_e5_ccfa"] is False
    assert len(report["gaps"]) >= 5
    assert all(row["status"] == "open" for row in report["gaps"])

