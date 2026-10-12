from datetime import datetime

import bench.e5.validate as validate
from bench.e5.validate import _unexpected_worktree, _valid_report_time, _verification_worktree_clean, build_evidence_manifest


def test_e5_validator_allows_authorized_e4b_and_e5_scope():
    status = (
        " M README.md\n M bench/e1/source.py\n?? bench/e4/cross_hardware.py\n"
        "?? results/e4b_machine_4090.json\n?? artifact/\n?? bench/e5/\n"
        "?? results/e5_regression.json\n?? test/test_e5_validate.py"
    )
    assert _unexpected_worktree(status) == []
    assert _unexpected_worktree(status + "\n M pact/e2_guard.py") == ["pact/e2_guard.py"]


def test_report_time_requires_timezone_and_valid_iso_string():
    now = datetime.now().astimezone()
    assert _valid_report_time(now.isoformat(), now)
    assert not _valid_report_time("2026-10-08T10:00:00", now)
    assert not _valid_report_time("not-a-time", now)


def test_final_clean_check_only_allows_proven_catalog_eol_diff(monkeypatch):
    monkeypatch.setattr(validate, "_catalog_has_eol_only_diff", lambda: True)
    assert _verification_worktree_clean("")
    assert _verification_worktree_clean(" M bench/catalog.json")
    assert not _verification_worktree_clean(" M bench/catalog.json\n M results/e5_regression.json")

    monkeypatch.setattr(validate, "_catalog_has_eol_only_diff", lambda: False)
    assert not _verification_worktree_clean(" M bench/catalog.json")


def test_formal_evidence_manifest_is_complete_and_content_addressed():
    report = build_evidence_manifest()
    assert all(report["checks"].values())
    assert all(row["sha256"] and len(row["sha256"]) == 64 for row in report["rows"])


def test_final_markdown_reports_true_go_without_pending_rows(tmp_path, monkeypatch):
    """最终冻结报告必须与 JSON 的 Go 状态一致，不能遗留开发期硬编码。"""

    monkeypatch.setattr(validate, "RESULTS", tmp_path)
    report = {
        "status": "complete",
        "go_e5_preparation": True,
        "go_e5_ccfb_readiness": True,
        "go_e5": True,
        "source_fingerprint": "a" * 64,
        "checks": {"final": True},
        "pending": [],
        "claims": {"readiness": "九项内部证据条件均已满足。"},
        "regression": {
            "command": ["python", "-m", "bench.e5.validate"],
            "exit_code": 0,
            "pytest_summary": "1 passed",
            "legacy_checks": {"case": True},
        },
    }

    validate.write(report)
    markdown = (tmp_path / "e5_regression.md").read_text(encoding="utf-8")
    assert "`go_e5=true`" in markdown
    assert "九项内部证据条件全部满足" in markdown

