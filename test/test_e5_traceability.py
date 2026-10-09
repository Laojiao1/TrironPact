from bench.e5.traceability import build


def test_every_e5_claim_has_existing_evidence_and_a_limit():
    report = build()
    assert report["go_traceability"], [row for row in report["rows"] if not row["complete"]]
    assert report["counts"]["claims"] >= 9
    assert all(row["evidence_level"] and row["limitations"] for row in report["rows"])


def test_traceability_distinguishes_static_runtime_and_empirical_evidence():
    levels = {row["evidence_level"] for row in build()["rows"]}
    assert "static_supported_subset" in levels
    assert "executable_invariant" in levels
    assert "bounded_empirical" in levels
    assert "machine_audit" in levels

