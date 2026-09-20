"""核对契约 DSL 的结构校验、三值判定和来源序列化。"""

import json

import pytest

from pact.contract_dsl import Condition, ContractClause, Origin, TensorMetadata, ValueRef


def _ref(kind: str, tensor: str = "X", axis: int | None = None, value: int | None = None) -> ValueRef:
    return ValueRef(kind, tensor="" if kind == "constant" else tensor, axis=axis, value=value)


def test_singleton_stride_condition_and_roundtrip():
    condition = Condition("or", parts=(
        Condition("eq", _ref("size", axis=0), _ref("constant", value=1)),
        Condition("eq", _ref("stride", axis=0), _ref("size", axis=1)),
    ))
    clause = ContractClause(condition, "Semantics", "Statically-Proven", ("CUDA float32", "二维线性读取"), Origin("scenarios/kernels.py", 15, "load", "X", "x[row, col]", (("N", "x.size(1)"),), "单行时行步长不参与读取地址"))
    metadata = TensorMetadata((1, 11), (14, 1), 0, 0x1000, 4, 56)
    assert clause.condition.evaluate({"X": metadata}) is True
    assert clause.condition.evaluate({"X": TensorMetadata((7, 11), (14, 1), 0, 0x1000, 4, 392)}) is False
    assert clause.condition.evaluate({}) is None
    restored = ContractClause.from_dict(json.loads(json.dumps(clause.to_dict())))
    assert restored == clause


def test_alignment_and_span_are_only_represented_until_proven():
    aligned = Condition("mod_eq", _ref("effective_ptr"), divisor=16, remainder=0)
    span = Condition("span_in_storage", tensor="X")
    assert aligned.evaluate({}) is None
    assert aligned.evaluate({"X": TensorMetadata((2, 3), (5, 1), 1, 0x1004, 4, 36)}) is False
    assert span.evaluate({"X": TensorMetadata((2, 3), (5, 1), 1, 0x1004, 4, 36)}) is True
    assert span.evaluate({"X": TensorMetadata((2, 3), (5, 1), 1, 0x1004, 4, 28)}) is False
    clause = ContractClause(aligned, "Optimization", "Unproven", ("待分析提示表达式",), Origin("fixture.py", 1, "load", "X", "x[i]", (), "尚未分析 tl.multiple_of 的作用对象"))
    assert clause.evidence == "Unproven"


def test_invalid_fields_are_rejected():
    with pytest.raises(ValueError):
        ValueRef("stride", tensor="X")
    with pytest.raises(ValueError):
        Condition("mod_eq", _ref("stride", axis=0), divisor=0, remainder=0)
    with pytest.raises(ValueError):
        Condition("and", parts=(Condition("span_in_storage", tensor="X"),))
    with pytest.raises(ValueError):
        ContractClause(Condition("span_in_storage", tensor="X"), "Semantics", "Unproven", (), Origin("a.py", 1, "load", "X", "x[i]", (), "检查来源"))
