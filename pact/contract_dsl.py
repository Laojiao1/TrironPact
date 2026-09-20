"""用有类型的条件树表示物理布局契约；本模块不负责从 Kernel 推导条件。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


Truth = bool | None  # None 表示当前元数据不足，不能据此放行 Fast。
ValueKind = Literal["constant", "size", "stride", "storage_offset", "effective_ptr", "element_bytes", "storage_nbytes", "scalar"]
Purpose = Literal["Safety", "Semantics", "Optimization"]
Evidence = Literal["Unproven", "Statically-Proven", "Empirically-Validated", "Unknown"]


@dataclass(frozen=True)
class TensorMetadata:
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    storage_offset: int
    effective_ptr: int
    element_bytes: int
    storage_nbytes: int


@dataclass(frozen=True)
class ValueRef:
    kind: ValueKind
    tensor: str = ""
    axis: int | None = None
    value: int | None = None
    scalar: str = ""

    def __post_init__(self) -> None:
        kinds = {"constant", "size", "stride", "storage_offset", "effective_ptr", "element_bytes", "storage_nbytes", "scalar"}
        if self.kind not in kinds:
            raise ValueError(f"未知属性类型：{self.kind}")
        if self.kind == "constant":
            if type(self.value) is not int or self.tensor or self.axis is not None or self.scalar:
                raise ValueError("常量只能填写整数 value")
        elif self.kind == "scalar":
            if not self.scalar or self.tensor or self.axis is not None or self.value is not None:
                raise ValueError("标量只需填写 scalar 名称")
        elif not self.tensor or self.value is not None or self.scalar:
            raise ValueError("Tensor 属性必须填写 tensor，不能附带常量或标量")
        elif self.kind in ("size", "stride"):
            if type(self.axis) is not int or self.axis < 0:
                raise ValueError("size/stride 需要非负维度")
        elif self.axis is not None:
            raise ValueError("此 Tensor 属性不接受维度")

    @classmethod
    def from_dict(cls, data: dict) -> ValueRef:
        return cls(**data)

    def read(self, tensors: dict[str, TensorMetadata], scalars: dict[str, int]) -> int | None:
        if self.kind == "constant":
            return self.value
        if self.kind == "scalar":
            return scalars.get(self.scalar)
        meta = tensors.get(self.tensor)
        if meta is None:
            return None
        if self.kind in ("size", "stride"):
            values = meta.shape if self.kind == "size" else meta.stride
            return values[self.axis] if self.axis < len(values) else None
        return getattr(meta, self.kind)


@dataclass(frozen=True)
class Condition:
    kind: Literal["eq", "mod_eq", "span_in_storage", "and", "or"]
    left: ValueRef | None = None
    right: ValueRef | None = None
    divisor: int | None = None
    remainder: int | None = None
    tensor: str = ""
    parts: tuple[Condition, ...] = ()

    def __post_init__(self) -> None:
        if self.kind == "eq":
            valid = self.left is not None and self.right is not None and self.divisor is None and self.remainder is None and not self.tensor and not self.parts
        elif self.kind == "mod_eq":
            valid = self.left is not None and self.right is None and type(self.divisor) is int and self.divisor > 0 and type(self.remainder) is int and 0 <= self.remainder < self.divisor and not self.tensor and not self.parts
        elif self.kind == "span_in_storage":
            valid = bool(self.tensor) and self.left is None and self.right is None and self.divisor is None and self.remainder is None and not self.parts
        elif self.kind in ("and", "or"):
            valid = len(self.parts) >= 2 and all(isinstance(item, Condition) for item in self.parts) and self.left is None and self.right is None and self.divisor is None and self.remainder is None and not self.tensor
        else:
            valid = False
        if not valid:
            raise ValueError(f"条件字段不匹配：{self.kind}")

    @classmethod
    def from_dict(cls, data: dict) -> Condition:
        values = dict(data)
        for key in ("left", "right"):
            if values.get(key) is not None:
                values[key] = ValueRef.from_dict(values[key])
        if "parts" in values:
            values["parts"] = tuple(cls.from_dict(item) for item in values["parts"])
        return cls(**values)

    def evaluate(self, tensors: dict[str, TensorMetadata], scalars: dict[str, int] | None = None) -> Truth:
        scalars = scalars or {}
        if self.kind in ("eq", "mod_eq"):
            left = self.left.read(tensors, scalars)
            if left is None:
                return None
            if self.kind == "mod_eq":
                return left % self.divisor == self.remainder
            right = self.right.read(tensors, scalars)
            return None if right is None else left == right
        if self.kind == "span_in_storage":
            meta = tensors.get(self.tensor)
            if meta is None or len(meta.shape) != len(meta.stride) or not meta.shape or any(size < 1 for size in meta.shape) or any(step < 0 for step in meta.stride) or meta.element_bytes < 1:
                return None
            last = meta.storage_offset + sum((size - 1) * step for size, step in zip(meta.shape, meta.stride))
            return meta.storage_offset >= 0 and (last + 1) * meta.element_bytes <= meta.storage_nbytes
        results = [part.evaluate(tensors, scalars) for part in self.parts]
        if self.kind == "and":
            return False if False in results else (None if None in results else True)
        return True if True in results else (None if None in results else False)


@dataclass(frozen=True)
class Origin:
    source_file: str
    source_line: int
    access_kind: Literal["load", "store"]
    pointer: str
    semantic_index: str
    bindings: tuple[tuple[str, str], ...]
    explanation: str

    def __post_init__(self) -> None:
        if not self.source_file or self.source_line < 1 or self.access_kind not in ("load", "store") or not self.pointer or not self.semantic_index or not self.explanation:
            raise ValueError("契约来源必须指向访存点和独立语义假设")


@dataclass(frozen=True)
class ContractClause:
    condition: Condition
    purpose: Purpose
    evidence: Evidence
    domain: tuple[str, ...]
    origin: Origin

    def __post_init__(self) -> None:
        if self.purpose not in ("Safety", "Semantics", "Optimization") or self.evidence not in ("Unproven", "Statically-Proven", "Empirically-Validated", "Unknown") or not self.domain:
            raise ValueError("契约用途、证据或适用域无效")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ContractClause:
        values = dict(data)
        values["condition"] = Condition.from_dict(values["condition"])
        origin = dict(values["origin"])
        origin["bindings"] = tuple(tuple(item) for item in origin["bindings"])
        values["origin"] = Origin(**origin)
        values["domain"] = tuple(values["domain"])
        return cls(**values)
