from pact.candidates import Candidate
from pact.contract_dsl import Condition, Origin, ValueRef
from pact.smt_refine import Domain, check_redundancy


ORIGIN = Origin("sample.py", 1, "load", "X", "x[i]", (("X", "x"),), "test")
REF = ValueRef("effective_ptr", tensor="X")
SCOPE = ("CUDA float32", "N >= 1")


def candidate(condition, purpose="Optimization", scope=SCOPE, evidence="Statically-Proven"):
    return Candidate(condition, None, purpose, evidence, ORIGIN, "test", scope, (("X", "x"),), "test")


def test_mod_implication_and_reverse_nonimplication():
    strong = candidate(Condition("mod_eq", left=REF, divisor=16, remainder=0))
    weak = candidate(Condition("mod_eq", left=REF, divisor=4, remainder=0))
    result = check_redundancy([strong, weak], Domain((("X", 1),)))
    assert result["status"] == "Supported" and result["removed"] == [1]
    reverse = check_redundancy([weak, strong], Domain((("X", 1),)))
    assert any(item["status"] == "sat" and item.get("candidate") == 1 for item in reverse["queries"])


def test_cross_purpose_and_domain_never_removed():
    condition = Condition("mod_eq", left=REF, divisor=4, remainder=0)
    result = check_redundancy([candidate(condition), candidate(condition, purpose="Safety"), candidate(condition, scope=("other",))], Domain((("X", 1),)))
    assert result["removed"] == []


def test_disjunction_conflict_and_unsupported_are_conservative():
    size = ValueRef("size", tensor="X", axis=0)
    stride = ValueRef("stride", tensor="X", axis=0)
    singleton = Condition("or", parts=(Condition("eq", left=size, right=ValueRef("constant", value=1)), Condition("eq", left=stride, right=ValueRef("constant", value=1))))
    assert check_redundancy([candidate(singleton), candidate(Condition("eq", left=stride, right=ValueRef("constant", value=1)))], Domain((("X", 1),)))["removed"] == [0]
    conflicting = Domain((("X", 1),), facts=(Condition("eq", left=size, right=ValueRef("constant", value=0)),))
    assert check_redundancy([candidate(singleton)], conflicting)["status"] == "Conflict"
    unsupported = candidate(Condition("span_in_storage", tensor="X"))
    assert check_redundancy([unsupported, unsupported], Domain((("X", 1),)))["removed"] == []


def test_solver_unknown_keeps_every_condition(monkeypatch):
    from pact import smt_refine

    monkeypatch.setattr(smt_refine, "_check", lambda terms, timeout_ms: ("unknown", "timed out", {}))
    value = candidate(Condition("mod_eq", left=REF, divisor=4, remainder=0))
    result = check_redundancy([value, value], Domain((("X", 1),)))
    assert result["status"] == "Unknown" and result["kept"] == [0, 1] and result["removed"] == []
