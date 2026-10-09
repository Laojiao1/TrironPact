from pathlib import Path

from pact.e2_guard import E2GuardResult


PROJECT = Path(__file__).resolve().parents[1]


def test_correctness_note_states_supported_domain_and_evidence_boundaries():
    text = (PROJECT / "artifact/e5_correctness.md").read_text(encoding="utf-8")
    for heading in (
        "受支持子语言与外部前提",
        "Access IR 语义",
        "候选充分性",
        "Guard 与分派",
        "Relayout 与二次复验",
        "指纹与过期证据",
        "声明边界",
    ):
        assert heading in text
    assert "data_ptr()" in text and "storage_offset()" in text
    assert "False/Unknown" in text and "不授予最终 readiness Go" in text


def test_guard_result_only_allows_explicit_true():
    assert E2GuardResult("True", "ok", ()).allowed
    assert not E2GuardResult("False", "rejected", ()).allowed
    assert not E2GuardResult("Unknown", "missing", ()).allowed


def test_relayout_source_requires_new_allocation_copy_and_recheck():
    text = (PROJECT / "pact/relayout.py").read_text(encoding="utf-8")
    assert "torch.empty(" in text
    assert ".copy_(source)" in text
    assert "guard = plan.evaluate(current)" in text
    assert text.index("torch.empty(") < text.index(".copy_(source)") < text.index("guard = plan.evaluate(current)")

