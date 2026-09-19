"""单元测试，涵盖 AST 提取、错误拦截、直通、修订和双输入测试"""

from pathlib import Path
from types import SimpleNamespace

import triton.language as tl

from pact.analysis import extract
from pact.oracle import run_isolated
from pact.refine import refine_singleton_strides
from pact.runtime import generated_contract, guard
from scenarios.cases import reference
from scenarios.kernels import KERNELS


def unsupported_hint_kernel(X, Y, N: tl.constexpr, B: tl.constexpr):
    """只用于验证尚未建模的优化提示不能进入 Fast，不执行该函数。"""
    idx = tl.program_id(0) * B + tl.arange(0, B)
    hint = tl.multiple_of(idx, 16)
    mask = idx < N
    value = tl.load(X + idx, mask=mask, other=0)
    tl.store(Y + idx, value + 1, mask=mask)


def test_access_ir_contains_real_addresses_and_masks():
    for name in ("A", "A2", "B"):
        result = extract(name)
        assert result.status == "Supported", result.reason
        assert len(result.accesses) == 2
        assert all("program_id" in access.index_calls and "arange" in access.index_calls for access in result.accesses)
        assert result.mask_excludes_tail
    a2 = extract("A2")
    assert "S0" in a2.accesses[0].expanded_offset
    assert [item.expected for item in a2.predicates] == ["1"]
    assert not any("%" in item.expected for item in extract("B").predicates)


def test_semantics_and_every_predicate_have_auditable_sources():
    for name in ("A", "A2", "B", "D"):
        result = extract(name)
        meaning = result.semantics
        assert meaning.reference_symbol == f"{reference.__module__}.{reference.__name__}"
        assert len(meaning.axes) == result.semantics.ndim
        assert meaning.index_domain and meaning.logical_write and meaning.launch_bindings
        for predicate in result.predicates:
            expected = meaning.input_for(predicate.tensor)
            assert predicate.semantic_input == expected.pointer
            assert predicate.semantic_index == expected.logical_read
            assert any(access.kind == "load" and access.tensor == expected.pointer and access.line == predicate.source_access_line and access.source_file == predicate.source_access_file for access in result.accesses)
            source_line = (Path(__file__).resolve().parent.parent / predicate.source_access_file).read_text(encoding="utf-8").splitlines()[predicate.source_access_line - 1]
            assert "tl.load(" in source_line
    assert ("S0", "x.stride(0)") in extract("A2").semantics.launch_bindings
    assert len(extract("D").semantics.inputs) == 2


def test_wrong_stride_is_observed_then_blocked():
    raw = run_isolated("A", "transpose", mode="raw")
    guarded = run_isolated("A", "transpose")
    assert raw.category == "numeric_mismatch", raw.detail
    assert raw.detail["mismatch_count"] > 0
    assert guarded.category == "correct", guarded.detail
    assert guarded.detail["path"] == "PyTorch Fallback"
    assert raw.evidence["observation_level"] == "Empirically-Validated"
    assert raw.evidence["fast_eligible"] is False
    assert guarded.evidence["fast_eligible"] is False


def test_padded_rows_run_fast_without_materialization():
    result = run_isolated("A2", "padded")
    assert result.category == "correct", result.detail
    assert result.detail["path"] == "Fast"
    assert result.detail["is_contiguous"] is False
    assert result.detail["contiguous_copied"] is True
    assert result.detail["input_ptr_unchanged"] is True


def test_masked_tail_and_unknown_input():
    for n in (127, 129):
        result = run_isolated("B", "contiguous", n=n)
        assert result.category == "correct", result.detail
        assert result.detail["path"] == "Fast"
        assert result.evidence["guard_basis"] == "Statically-Proven"
        assert result.evidence["fast_eligible"] is True
        assert result.evidence["observation_level"] == "Empirically-Validated"
    unknown = run_isolated("A", "unknown_dtype")
    assert unknown.category == "correct", unknown.detail
    assert unknown.detail["path"] == "PyTorch Fallback"
    assert unknown.detail["guard"]["evidence"] == "Unknown"
    assert unknown.evidence["guard_basis"] == "Unknown"
    assert unknown.evidence["observation_level"] == "Empirically-Validated"
    assert unknown.evidence["fast_eligible"] is False


def test_boundary_evidence_refines_overstrong_stride_conditions():
    row = run_isolated("A", "single_row", m=1, n=11, mode="raw").to_dict()
    col = run_isolated("A", "single_col", m=7, n=1, mode="raw").to_dict()
    initial = extract("A")
    unchanged, incomplete = refine_singleton_strides(initial, [row])
    assert unchanged.predicates == initial.predicates
    assert incomplete["changed"] is False
    revised, evidence = refine_singleton_strides(initial, [row, col])
    assert evidence["changed"] is True
    assert [item.unless_size_one for item in revised.predicates] == [0, 1]
    for layout, m, n in (("single_row", 1, 11), ("single_col", 7, 1)):
        before = run_isolated("A", layout, m=m, n=n)
        after = run_isolated("A", layout, m=m, n=n, refined=True)
        assert before.category == after.category == "correct"
        assert before.detail["path"] == "PyTorch Fallback"
        assert after.detail["path"] == "Fast"


def test_official_vector_add_extracts_two_input_conditions():
    extracted = extract("D")
    assert extracted.status == "Supported", extracted.reason
    assert [item.tensor for item in extracted.predicates] == ["X", "Y"]
    assert len(extracted.accesses) == 3
    correct = run_isolated("D", "contiguous", n=129)
    wrong = run_isolated("D", "x_strided", n=129, mode="raw")
    blocked = run_isolated("D", "x_strided", n=129)
    assert correct.category == "correct" and correct.detail["path"] == "Fast"
    assert wrong.category == "numeric_mismatch"
    assert blocked.category == "correct" and blocked.detail["path"] == "PyTorch Fallback"


def test_unmodeled_hint_is_unsupported_and_cannot_enter_fast(monkeypatch):
    # 临时替换 B 的源码入口，检查拒绝路径；测试结束后清除契约缓存。
    monkeypatch.setitem(KERNELS, "B", SimpleNamespace(fn=unsupported_hint_kernel))
    generated_contract.cache_clear()
    try:
        analysis = extract("B")
        assert analysis.status == "Unsupported"
        assert "multiple_of" in analysis.reason
        decision = guard("B", object())
        assert decision.allowed is False
        assert decision.evidence == "Unsupported"
    finally:
        generated_contract.cache_clear()
