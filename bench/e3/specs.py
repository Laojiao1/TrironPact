"""e3 风险案例与上游防御证据的冻结规格。"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RiskSpec:
    """描述一个允许独立执行的真实 Kernel 物理布局风险案例。

    ``mutated_inputs`` 只允许 worker 已实现的合法 PyTorch 视图。L1 表示底层
    Kernel 前置条件敏感性；只有 ``wrapper_defense`` 指向固定上游源码时才
    同时形成 L2。L3 必须引用公开 Issue/PR，不能由本地复现自动推断。
    """

    id: str
    kernel_id: str
    source: str
    problem: str
    mutated_inputs: tuple[str, ...]
    mutation: str
    wrapper_defense: str | None
    wrapper_source: str | None
    l3_url: str | None = None
    l3_status: str | None = None
    l3_commit: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


RISK_SPECS = (
    RiskSpec(
        "triton_add_x_stride",
        "triton_add",
        "triton",
        "一维输入列步长未进入地址表达式",
        ("x",),
        "column_stride_2",
        None,
        None,
    ),
    RiskSpec(
        "liger_softmax_x_stride",
        "liger_softmax",
        "liger",
        "二维输入只传行步长，Kernel 假设列步长为 1",
        ("x",),
        "column_stride_2",
        "x.contiguous().view(-1, n_cols)",
        "bench/e1/upstream/liger/src/liger_kernel/ops/softmax.py:118",
    ),
    RiskSpec(
        "unsloth_layernorm_x_stride",
        "unsloth_layernorm",
        "unsloth",
        "LayerNorm 输入列步长未传入 Kernel",
        ("x",),
        "column_stride_2",
        "X.reshape(-1, dim).contiguous()",
        "bench/e1/upstream/unsloth/unsloth/kernels/layernorm.py:107",
        "https://github.com/unslothai/unsloth/pull/10675",
        "open_at_2026-10-02",
        "1d254f8c54",
    ),
    RiskSpec(
        "unsloth_layernorm_affine_stride",
        "unsloth_layernorm",
        "unsloth",
        "LayerNorm W/b 使用单位列偏移且没有 stride 参数",
        ("w", "b"),
        "column_stride_2",
        "W.contiguous(); b.contiguous()",
        "bench/e1/upstream/unsloth/unsloth/kernels/layernorm.py:109-110",
        "https://github.com/unslothai/unsloth/pull/10675",
        "open_at_2026-10-02",
        "624d9b86d",
    ),
)


def risk_spec(case_id: str) -> RiskSpec:
    for item in RISK_SPECS:
        if item.id == case_id:
            return item
    raise ValueError(f"Unknown：未登记的 e3 风险案例 {case_id}")

