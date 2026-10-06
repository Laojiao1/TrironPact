from bench.e4.baselines import POLICIES, build


def test_e4_baselines_share_denominator_and_disclose_unsafe_errors():
    report = build()
    assert report["go_baselines"]
    rows = {row["policy"]: row for row in report["rows"]}
    assert tuple(rows) == POLICIES
    assert rows["static_no_refinement"]["errors"] == 2
    assert rows["tritonpact"]["errors"] == 0
    assert rows["always_copy"]["copy_bytes"] > rows["upstream"]["copy_bytes"] > 0
    assert all(row["correct"] + row["errors"] == report["case_domain"]["total"] for row in rows.values())
