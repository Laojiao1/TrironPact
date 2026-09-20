"""验证统一 IR 到 PoC 候选的绑定边界，以及拒绝路径不会获得 Fast 资格。"""

from dataclasses import replace

from pact.analysis import extract
from pact.runtime import generated_contract, guard
from pact.semantics import SEMANTICS
from pact.stage_report import build_stage_report, render_stage_report


def test_poc_predicates_have_typed_sources_and_refinement():
    for name in ("A", "A2", "B", "D"):
        analysis = extract(name)
        assert analysis.status == "Supported", analysis.reason
        assert len(analysis.clauses) == len(analysis.predicates)
        assert all(clause.purpose == "Semantics" and clause.origin.source_line > 0 for clause in analysis.clauses)
        assert all(clause.origin.semantic_index and clause.domain for clause in analysis.clauses)
    generated_contract.cache_clear()
    revised = generated_contract("A", refined=True)
    assert all(clause.condition.kind == "or" for clause in revised.clauses)


def test_missing_semantics_or_wrong_bindings_cannot_enter_fast(monkeypatch):
    original = SEMANTICS["A2"]
    monkeypatch.delitem(SEMANTICS, "A2")
    generated_contract.cache_clear()
    try:
        assert extract("A2").status == "Unknown"
        decision = guard("A2", object())
        assert not decision.allowed and decision.evidence == "Unknown"
    finally:
        generated_contract.cache_clear()
    monkeypatch.setitem(SEMANTICS, "A2", replace(original, launch_bindings=tuple((name, "x.stride(1)" if name == "S0" else value) for name, value in original.launch_bindings)))
    generated_contract.cache_clear()
    try:
        analysis = extract("A2")
        assert analysis.status == "Unknown" and "绑定" in analysis.reason
        decision = guard("A2", object())
        assert not decision.allowed and decision.evidence == "Unknown"
    finally:
        generated_contract.cache_clear()


def test_wrong_output_or_computation_is_rejected(monkeypatch):
    original = SEMANTICS["B"]
    monkeypatch.setitem(SEMANTICS, "B", replace(original, output_pointer="missing_output"))
    assert extract("B").status == "Unknown"
    monkeypatch.setitem(SEMANTICS, "B", replace(original, logical_write="out[i] = x[i] + 2"))
    result = extract("B")
    assert result.status == "Unknown"
    assert "输出计算" in result.reason


def test_stage_report_separates_ir_from_poc_contract():
    report = build_stage_report()
    assert report["go_ir"] is True
    assert all(item["ir"]["status"] == "Supported" for item in report["cases"].values())
    assert "不授予 Fast 资格" in render_stage_report(report)
