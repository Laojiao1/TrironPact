from bench.e5.holdout import build


def test_e5_holdout_is_read_only_audit_of_frozen_result():
    report = build()
    assert report["go_holdout_audit"]
    assert report["mode"] == "read_only_audit_of_e2_frozen_holdout_v1"
    assert report["rerun_performed"] is False
    assert report["rules_changed_since_evaluation"] is False
    assert report["counts"] == {
        "holdout": 7,
        "supported": 6,
        "unknown": 1,
        "fast_feasible": 6,
        "known_false_allows": 0,
    }
    assert [row["id"] for row in report["rows"] if row["access_status"] == "Unknown"] == ["flag_slice"]

