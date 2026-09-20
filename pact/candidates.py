"""第三阶段的离线候选、义务和受限调用绑定；不参与运行时分派。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from pact.access_ir import Expr, ParseResult
from pact.contract_dsl import Condition, Origin, TensorMetadata
from pact.semantics import OperatorMeaning


def _fields(data: dict, expected: set[str]) -> None:
    if not isinstance(data, dict) or set(data) != expected:
        raise ValueError(f"字段不匹配：期望 {sorted(expected)}，得到 {sorted(data) if isinstance(data, dict) else type(data).__name__}")


@dataclass(frozen=True)
class Candidate:
    """condition 是可求值谓词；obligation 是暂不能安全降为谓词的义务。"""

    condition: Condition | None
    obligation: Expr | None
    purpose: Literal["Safety", "Semantics", "Optimization"]
    evidence: Literal["Statically-Proven", "Unproven", "Unknown"]
    origin: Origin
    rule: str
    domain: tuple[str, ...]
    binding: tuple[tuple[str, str], ...]
    reason: str

    def __post_init__(self) -> None:
        if (self.condition is None) == (self.obligation is None):
            raise ValueError("候选必须恰有一个条件或未降级义务")
        if self.purpose not in ("Safety", "Semantics", "Optimization") or self.evidence not in ("Statically-Proven", "Unproven", "Unknown"):
            raise ValueError("候选用途或证据状态无效")
        if not self.rule or not self.domain or not self.binding or not self.reason:
            raise ValueError("候选缺少推导规则、支持域、绑定或理由")
        if self.evidence == "Statically-Proven" and self.condition is None:
            raise ValueError("未降级义务不能标记为静态充分")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> Candidate:
        _fields(data, set(cls.__dataclass_fields__))
        values = dict(data)
        if values["condition"] is not None:
            values["condition"] = Condition.from_dict(values["condition"])
        if values["obligation"] is not None:
            values["obligation"] = _expr_from_dict(values["obligation"])
        origin = values["origin"]
        _fields(origin, set(Origin.__dataclass_fields__))
        values["origin"] = Origin(**{**origin, "bindings": tuple(tuple(pair) for pair in origin["bindings"])})
        values["domain"] = tuple(values["domain"])
        values["binding"] = tuple(tuple(pair) for pair in values["binding"])
        return cls(**values)

    def evaluate(self, tensors: dict[str, TensorMetadata], scalars: dict[str, int] | None = None) -> bool | None:
        """运行时求值只供影子报告；证据等级和 Fast 资格不受结果影响。"""
        return self.condition.evaluate(tensors, scalars) if self.condition is not None else None


def _expr_from_dict(data: dict) -> Expr:
    _fields(data, {"op", "value", "args"})
    return Expr(data["op"], data["value"], tuple(_expr_from_dict(part) for part in data["args"]))


@dataclass(frozen=True)
class CallBinding:
    """由调用处独立提供的受限启动事实；validate 不解析任意 wrapper。"""

    pointers: tuple[tuple[str, str], ...]
    scalars: tuple[tuple[str, str], ...]
    grid: str
    output_shape: tuple[str, ...]
    output_element_bytes: int | None
    source: str

    def __post_init__(self) -> None:
        if not self.source or len(dict(self.pointers)) != len(self.pointers) or len(dict(self.scalars)) != len(self.scalars):
            raise ValueError("调用来源或参数绑定无效")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> CallBinding:
        _fields(data, set(cls.__dataclass_fields__))
        return cls(tuple(tuple(pair) for pair in data["pointers"]), tuple(tuple(pair) for pair in data["scalars"]), data["grid"], tuple(data["output_shape"]), data["output_element_bytes"], data["source"])


@dataclass(frozen=True)
class BindingResult:
    status: Literal["Supported", "Unknown", "Unsupported"]
    reasons: tuple[str, ...]


def validate_binding(ir: ParseResult, meaning: OperatorMeaning | None, call: CallBinding) -> BindingResult:
    """核对声明之间的一致性；source 必须由调用处另行核验，不能自证。"""
    if ir.status != "Supported":
        return BindingResult(ir.status, (ir.reason,))
    if meaning is None or not meaning.inputs or not meaning.index_domain or not meaning.logical_write or not meaning.reference_symbol:
        return BindingResult("Unknown", ("缺少独立算子语义",))
    reasons = []
    pointers = {item.pointer: item.tensor for item in meaning.inputs}
    pointers[meaning.output_pointer] = "OUT"
    if dict(call.pointers) != pointers or set(pointers) - set(ir.signature):
        reasons.append("指针身份与独立语义或 Kernel 签名冲突")
    declared = dict(meaning.launch_bindings)
    expected_scalars = {key: value for key, value in declared.items() if key != "grid"}
    if dict(call.scalars) != expected_scalars or set(expected_scalars) != set(ir.signature) - set(pointers):
        reasons.append("shape/stride/Tile 标量与独立语义或 Kernel 签名冲突")
    grid = declared.get("grid")
    valid_grids = {"M", "ceil(N/B)", "ceil(M*N/B)", "ceil(n_elements/BLOCK_SIZE)"}
    if call.grid != grid or grid not in valid_grids:
        reasons.append("grid 未匹配受支持的调用绑定")
    shape = meaning.inputs[0].shape_params
    if call.output_shape != shape or not shape or any(item.shape_params != shape for item in meaning.inputs):
        reasons.append("新建输出形状与独立逻辑形状不符")
    widths = {point.element_bytes for point in ir.accesses}
    if call.output_element_bytes is None or call.output_element_bytes < 1 or widths != {call.output_element_bytes}:
        reasons.append("输出元素宽度缺失或与访存点不一致")
    actual = [(point.kind, point.pointer_param, point.tensor) for point in ir.accesses]
    expected = [("load", item.pointer, item.tensor) for item in meaning.inputs] + [("store", meaning.output_pointer, "OUT")]
    if sorted(actual) != sorted(expected):
        reasons.append("load/store 与独立语义的指针访问不符")
    return BindingResult("Unknown" if reasons else "Supported", tuple(reasons))
