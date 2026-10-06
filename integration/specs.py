"""冻结 e4a 工作负载、调用位置和 Kernel 计数口径。"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class IntegrationStep:
    """一个实际进入工作负载轨迹的冻结 Kernel 函数体。

    ``call_site`` 描述 e4a 适配器中的逻辑调用位置，``upstream_role`` 说明
    上游真实使用方式。重复调用同一 ``kernel_id`` 不增加函数体计数。
    """

    kernel_id: str
    call_site: str
    upstream_role: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class WorkloadSpec:
    """描述可复现的真实调用轨迹；它不是完整模型吞吐基准。"""

    id: str
    title: str
    kind: str
    entrypoint: str
    input_domain: str
    provenance: str
    steps: tuple[IntegrationStep, ...]

    def to_dict(self) -> dict:
        return {**asdict(self), "steps": [item.to_dict() for item in self.steps]}


WORKLOADS = (
    WorkloadSpec(
        id="inductor_custom_kernel_graph",
        title="PyTorch Inductor 自定义 Triton Kernel 图",
        kind="compiler_graph_trace",
        entrypoint="integration.worker.run_inductor_workload",
        input_domain="固定 129 元素 pointwise 分支与 16x16 stride-aware 分支，float32 CUDA 张量",
        provenance="PyTorch test/inductor/test_triton_kernels.py 的 view、strided input 与多 Kernel 调用模式",
        steps=(
            IntegrationStep("pytorch_sub", "pointwise.sub", "PyTorch 自定义 Triton 二元算子"),
            IntegrationStep("pytorch_double", "pointwise.double", "PyTorch view/functionalization 测试 Kernel"),
            IntegrationStep("pytorch_masked_add", "pointwise.masked_add", "PyTorch bool mask 自定义 Kernel"),
            IntegrationStep("triton_add", "pointwise.add", "Triton 官方 vector-add Kernel 作为用户 Kernel 节点"),
            IntegrationStep("pytorch_double_strided", "layout.double_strided", "PyTorch stride-aware 二维 Kernel"),
        ),
    ),
    WorkloadSpec(
        id="transformer_layer_trace",
        title="Transformer 层归一化与门控激活轨迹",
        kind="model_layer_trace",
        entrypoint="integration.worker.run_transformer_workload",
        input_domain="固定小规模随机参数；8x80 归一化分支与 32x256 门控激活分支，float32 CUDA 张量",
        provenance="Liger/Unsloth 固定提交中的公开 wrapper、算子测试和独立 PyTorch reference",
        steps=(
            IntegrationStep("liger_rmsnorm", "norm.liger_rmsnorm", "Transformer RMSNorm"),
            IntegrationStep("unsloth_rmsnorm", "norm.unsloth_rmsnorm", "Llama RMSNorm wrapper"),
            IntegrationStep("liger_layernorm", "norm.liger_layernorm", "Transformer LayerNorm"),
            IntegrationStep("unsloth_layernorm", "norm.unsloth_layernorm", "Unsloth LayerNorm wrapper"),
            IntegrationStep("liger_swiglu", "activation.swiglu", "Transformer FFN SwiGLU"),
            IntegrationStep("liger_swiglu_tiled", "activation.swiglu_tiled", "宽行 SwiGLU 变体"),
            IntegrationStep("liger_geglu", "activation.geglu", "Transformer FFN GeGLU"),
            IntegrationStep("liger_fused_swiglu", "activation.fused_swiglu", "融合 gate/up SwiGLU"),
            IntegrationStep("liger_softmax", "activation.softmax", "行归约 softmax"),
        ),
    ),
)


def workload_spec(workload_id: str) -> WorkloadSpec:
    """返回冻结工作负载；未知 ID 不能隐式降级为微基准。"""

    for item in WORKLOADS:
        if item.id == workload_id:
            return item
    raise ValueError(f"Unknown：未登记的 e4a 工作负载 {workload_id}")


def distinct_kernel_ids() -> tuple[str, ...]:
    """按登记顺序返回不同函数体，重复调用不会增加 e4a 分母。"""

    return tuple(dict.fromkeys(step.kernel_id for workload in WORKLOADS for step in workload.steps))

