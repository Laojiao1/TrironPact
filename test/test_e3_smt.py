from bench.e3.smt import build


def test_real_smt_audit_preserves_denominator_and_scope():
    report = build(timeout_ms=2000)
    assert report["go_smt"]
    assert report["counts"]["kernels"] == 28
    assert report["counts"]["audited"] == 27
    assert report["counts"]["real_candidates"] == 158
    assert report["counts"]["synthetic_deletions"] == 0
    assert any(row["id"] == "flag_slice" and row["status"] == "Unknown" for row in report["rows"])

