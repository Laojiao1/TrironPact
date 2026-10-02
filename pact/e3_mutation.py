"""e3 公平变异域与确定性调度策略。

该模块只决定合法配方及其顺序，不读取 GPU 结果。实证结果由独立 worker
生成并以 case ID 绑定，避免让某种策略从 Oracle 结果反向挑选输入。
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from typing import Literal


MutationFamily = Literal[
    "multi_axis_grid",
    "stride_broadcast_2d",
    "separate_inputs",
    "offset_alignment",
    "span_boundary",
    "reduction_row_length",
]
Method = Literal["static_candidate", "uniform_random", "constrained_random", "predicate_guided"]


@dataclass(frozen=True)
class BoundaryProbe:
    """共享合法域中的一个候选边界。

    ``empirical_case`` 为 None 时只做元数据快筛。``executable=False`` 表示
    PyTorch 无法构造该物理视图，必须保留不可达理由，禁止送入 GPU worker。
    """

    id: str
    kernel_id: str
    source: str
    family: MutationFamily
    label: str
    changed_inputs: tuple[str, ...]
    predicate_flips: int
    empirical_case: str | None = None
    executable: bool = True
    unreachable_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.id or not self.kernel_id or not self.source or self.predicate_flips < 0:
            raise ValueError("变异 probe 字段无效")
        if self.executable == (self.unreachable_reason is not None):
            raise ValueError("不可执行 probe 必须且只能提供不可达理由")
        if self.empirical_case and not self.executable:
            raise ValueError("不可达 probe 不能绑定 GPU worker")

    @property
    def fingerprint(self) -> str:
        raw = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()


def probe_domain() -> tuple[BoundaryProbe, ...]:
    """返回冻结的共享输入域；每个新条件族至少有一项及不可达边界。"""

    return (
        BoundaryProbe("grid_edge", "pytorch_double_strided", "pytorch", "multi_axis_grid", "grid_axis1_last_program", ("x",), 1),
        BoundaryProbe("grid_mixed", "pytorch_double_strided", "pytorch", "multi_axis_grid", "two_axes_changed", ("x", "out"), 2),
        BoundaryProbe("triton_x_stride", "triton_add", "triton", "separate_inputs", "x_stride_2", ("x",), 1, "triton_add_x_stride"),
        BoundaryProbe("triton_offset", "triton_add", "triton", "offset_alignment", "x_offset_1", ("x",), 1),
        BoundaryProbe("triton_two_inputs", "triton_add", "triton", "separate_inputs", "x_y_stride_2", ("x", "y"), 2),
        BoundaryProbe("liger_row_padding", "liger_softmax", "liger", "stride_broadcast_2d", "row_padding", ("x",), 1),
        BoundaryProbe("liger_col_stride", "liger_softmax", "liger", "stride_broadcast_2d", "column_stride_2", ("x",), 2, "liger_softmax_x_stride"),
        BoundaryProbe("liger_row_minus", "liger_softmax", "liger", "reduction_row_length", "n_cols_minus_1", ("x", "out"), 3),
        BoundaryProbe("liger_row_plus", "liger_softmax", "liger", "reduction_row_length", "n_cols_plus_1", ("x", "out"), 3),
        BoundaryProbe("unsloth_x_stride", "unsloth_layernorm", "unsloth", "stride_broadcast_2d", "x_column_stride_2", ("x",), 2, "unsloth_layernorm_x_stride"),
        BoundaryProbe("unsloth_affine_stride", "unsloth_layernorm", "unsloth", "separate_inputs", "w_b_stride_2", ("w", "b"), 4, "unsloth_layernorm_affine_stride"),
        BoundaryProbe("unsloth_mixed", "unsloth_layernorm", "unsloth", "separate_inputs", "x_w_b_stride_2", ("x", "w", "b"), 6),
        BoundaryProbe("span_exact", "triton_add", "triton", "span_boundary", "last_enabled_element", ("x",), 1),
        BoundaryProbe(
            "span_one_short",
            "triton_add",
            "triton",
            "span_boundary",
            "storage_one_element_short",
            ("x",),
            1,
            executable=False,
            unreachable_reason="as_strided 要求的 storage 小于最大可访问元素，PyTorch 拒绝构造；仅作元数据反例",
        ),
    )


def ordered_probes(method: Method, seed: int) -> tuple[BoundaryProbe, ...]:
    """在相同冻结域上生成顺序；不得依据 GPU 结果排序。"""

    probes = list(probe_domain())
    rng = random.Random(seed)
    if method == "static_candidate":
        return tuple(probes)
    if method == "uniform_random":
        rng.shuffle(probes)
        return tuple(probes)
    if method == "constrained_random":
        groups: dict[str, list[BoundaryProbe]] = {}
        for probe in probes:
            groups.setdefault(probe.family, []).append(probe)
        for group in groups.values():
            rng.shuffle(group)
        ordered = []
        while any(groups.values()):
            for family in sorted(groups):
                if groups[family]:
                    ordered.append(groups[family].pop())
        return tuple(ordered)
    if method == "predicate_guided":
        # 只使用静态谓词翻转数；seed 仅打散同分项，不查看经验结果。
        rng.shuffle(probes)
        probes.sort(key=lambda item: (item.predicate_flips, item.executable), reverse=True)
        return tuple(probes)
    raise ValueError(f"Unknown：未知变异方法 {method}")

