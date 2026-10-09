from datetime import datetime

from bench.e5.validate import _unexpected_worktree, _valid_report_time, build_evidence_manifest


def test_e5_validator_allows_only_e5a_scope():
    status = "?? artifact/\n?? bench/e5/\n?? results/e5_regression.json\n?? test/test_e5_validate.py"
    assert _unexpected_worktree(status) == []
    assert _unexpected_worktree(status + "\n M pact/e2_guard.py") == ["pact/e2_guard.py"]


def test_report_time_requires_timezone_and_valid_iso_string():
    now = datetime.now().astimezone()
    assert _valid_report_time(now.isoformat(), now)
    assert not _valid_report_time("2026-10-08T10:00:00", now)
    assert not _valid_report_time("not-a-time", now)


def test_formal_evidence_manifest_is_complete_and_content_addressed():
    report = build_evidence_manifest()
    assert all(report["checks"].values())
    assert all(row["sha256"] and len(row["sha256"]) == 64 for row in report["rows"])

