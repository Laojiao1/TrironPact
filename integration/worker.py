"""在独立进程中执行 e4a 的两个冻结工作负载轨迹。"""

from __future__ import annotations

import argparse
import json
import time

import torch
import triton

from bench.e1.catalog import load
from bench.e1.specs import construct, reference, value
from bench.e1.worker import decode
from bench.e2.coverage import _kernel
from bench.e2.plans import build_plan
from integration.specs import workload_spec


_CATALOG = {item["id"]: item for item in load()["entries"]}
_INDUCTOR_KERNELS = {
    name: _kernel(_CATALOG[name])
    for name in ("pytorch_sub", "pytorch_double", "pytorch_masked_add", "triton_add", "pytorch_double_strided")
}


def _guard_row(kernel_id: str, tensors: dict[str, torch.Tensor]) -> dict:
    result = build_plan(_CATALOG[kernel_id]).evaluate(tensors)
    return {"kernel_id": kernel_id, "status": result.status, "allowed": result.allowed, "reason": result.reason, "trace": result.trace}


def _checks(entry: dict, tensors: dict[str, torch.Tensor], expected: tuple[torch.Tensor, ...]) -> list[dict]:
    spec = decode(entry["smoke"])
    rows = []
    for name, wanted in zip(spec.outputs, expected):
        actual = tensors[name]
        compatible = actual.shape == wanted.shape and actual.dtype == wanted.dtype
        close = torch.isclose(actual, wanted, rtol=spec.rtol, atol=spec.atol, equal_nan=False) if compatible else None
        rows.append(
            {
                "output": name,
                "correct": bool(compatible and close.all().item()),
                "mismatch_count": int((~close).sum().item()) if close is not None else None,
                "max_abs_error": float((actual.float() - wanted.float()).abs().max().item()) if compatible else None,
            }
        )
    return rows


def _execute_entry(kernel_id: str, seed: int, overrides: dict[str, torch.Tensor] | None = None) -> tuple[dict, torch.Tensor]:
    """执行一个冻结 wrapper 规格；Guard 非 True 时返回 Fallback 结果。

    override 仅替换同 shape/dtype 的数据流输入。它不改变 e2 冻结规则；任何
    shape/stride 偏离都会由现有 Guard 拒绝，并进入独立参考实现。
    """

    entry = _CATALOG[kernel_id]
    spec = decode(entry["smoke"])
    tensors = construct(spec, seed)
    for name, tensor in (overrides or {}).items():
        expected = tensors.get(name)
        if expected is None or expected.shape != tensor.shape or expected.dtype != tensor.dtype:
            raise ValueError(f"Unknown：{kernel_id} 的链式输入 {name} 与冻结规格不兼容")
        tensors[name] = tensor
    wanted = reference(spec, tensors)
    guard = build_plan(entry).evaluate(tensors)
    path = "Fast" if guard.allowed else "PyTorch Fallback"
    if guard.allowed:
        arguments = {binding.parameter: value(binding.expression, tensors) for binding in spec.bindings}
        kernel = _kernel(entry)
        if tuple(arguments) != tuple(kernel.arg_names):
            raise ValueError(f"Unknown：{kernel_id} 的 JIT 参数与冻结绑定不一致")
        kernel[spec.grid](**arguments, num_warps=4)
        torch.cuda.synchronize()
    else:
        for name, fallback in zip(spec.outputs, wanted):
            tensors[name].copy_(fallback)
    checks = _checks(entry, tensors, wanted)
    return (
        {
            "kernel_id": kernel_id,
            "path": path,
            "guard": {"status": guard.status, "allowed": guard.allowed, "reason": guard.reason, "trace": guard.trace},
            "checks": checks,
            "input_metadata": {
                name: {
                    "shape": list(tensors[name].shape),
                    "stride_elements": list(tensors[name].stride()),
                    "storage_offset_elements": tensors[name].storage_offset(),
                }
                for name in spec.inputs
            },
        },
        tensors[spec.outputs[0]],
    )


def _inductor_graph(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor, matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """复现 PyTorch 官方多 Kernel、view/stride 调用模式的受限图。"""

    sub = torch.empty_like(x)
    _INDUCTOR_KERNELS["pytorch_sub"][(2,)](in_ptr0=x, in_ptr1=y, out_ptr=sub, n_elements=x.numel(), BLOCK_SIZE=128)
    doubled = torch.empty_like(x)
    _INDUCTOR_KERNELS["pytorch_double"][(2,)](in_ptr0=sub, out_ptr=doubled, n_elements=x.numel(), BLOCK_SIZE=128)
    masked = torch.empty_like(x)
    _INDUCTOR_KERNELS["pytorch_masked_add"][(2,)](
        in_ptr0=doubled,
        in_ptr1=y,
        mask_ptr=mask,
        out_ptr=masked,
        n_elements=x.numel(),
        BLOCK_SIZE=128,
    )
    added = torch.empty_like(x)
    _INDUCTOR_KERNELS["triton_add"][(2,)](x_ptr=masked, y_ptr=y, output_ptr=added, n_elements=x.numel(), BLOCK_SIZE=128)
    matrix_out = torch.empty_like(matrix)
    _INDUCTOR_KERNELS["pytorch_double_strided"][(2, 2)](
        in_ptr=matrix,
        out_ptr=matrix_out,
        in_y_stride=matrix.stride(0),
        out_y_stride=matrix_out.stride(0),
        X_BLOCK_SIZE=8,
        Y_BLOCK_SIZE=8,
    )
    return added, matrix_out


def _inductor_inputs(seed: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    return (
        torch.randn(129, device="cuda", dtype=torch.float32),
        torch.randn(129, device="cuda", dtype=torch.float32),
        torch.rand(129, device="cuda") > 0.5,
        torch.randn((16, 16), device="cuda", dtype=torch.float32),
    )


def _inductor_guards(inputs: tuple[torch.Tensor, ...]) -> list[dict]:
    x, y, mask, matrix = inputs
    a = torch.empty_like(x)
    b = torch.empty_like(x)
    c = torch.empty_like(x)
    d = torch.empty_like(x)
    matrix_out = torch.empty_like(matrix)
    return [
        _guard_row("pytorch_sub", {"x": x, "y": y, "out": a}),
        _guard_row("pytorch_double", {"x": a, "out": b}),
        _guard_row("pytorch_masked_add", {"x": b, "y": y, "mask": mask, "out": c}),
        _guard_row("triton_add", {"x": c, "y": y, "out": d}),
        _guard_row("pytorch_double_strided", {"x": matrix, "out": matrix_out}),
    ]


def run_inductor_workload(seed: int = 17) -> dict:
    """运行真实 `torch.compile` 图；首次编译与稳态执行在报告中分开。"""

    workload = workload_spec("inductor_custom_kernel_graph")
    inputs = _inductor_inputs(seed)
    guards = _inductor_guards(inputs)
    if not all(row["allowed"] for row in guards):
        return {"workload": workload.id, "status": "Unknown", "reason": "frozen_guard_rejected_registered_input", "guards": guards}
    x, y, mask, matrix = inputs
    reference_vector = torch.where(mask, (x - y) * 2 + y, (x - y) * 2) + y
    reference_matrix = matrix * 2
    compiled = torch.compile(_inductor_graph, fullgraph=True, backend="inductor")
    started = time.perf_counter()
    vector, matrix_out = compiled(x, y, mask, matrix)
    torch.cuda.synchronize()
    first_run_ms = (time.perf_counter() - started) * 1000
    checks = [
        {"output": "vector", "correct": bool(torch.equal(vector, reference_vector)), "max_abs_error": float((vector - reference_vector).abs().max().item())},
        {"output": "matrix", "correct": bool(torch.equal(matrix_out, reference_matrix)), "max_abs_error": float((matrix_out - reference_matrix).abs().max().item())},
    ]
    return {
        "workload": workload.id,
        "status": "passed" if all(item["correct"] for item in checks) else "failed",
        "execution": "torch.compile(fullgraph=True, backend='inductor')",
        "first_run_including_compile_ms": first_run_ms,
        "guards": guards,
        "checks": checks,
        "kernel_ids": [step.kernel_id for step in workload.steps],
        "evidence": "真实 Inductor 自定义 Triton 图的冻结输入经验核对；不证明域外布局安全。",
    }


def run_transformer_workload(seed: int = 17) -> dict:
    """运行模型层数据流及上游 Kernel 变体，不加载预训练权重。"""

    workload = workload_spec("transformer_layer_trace")
    rows: list[dict] = []
    rms_row, rms_out = _execute_entry("liger_rmsnorm", seed)
    rows.append(rms_row)
    unsloth_rms_row, unsloth_rms_out = _execute_entry("unsloth_rmsnorm", seed + 1, {"x": rms_out})
    rows.append(unsloth_rms_row)
    layer_row, layer_out = _execute_entry("liger_layernorm", seed + 2, {"x": unsloth_rms_out})
    rows.append(layer_row)
    unsloth_layer_row, _ = _execute_entry("unsloth_layernorm", seed + 3, {"x": layer_out})
    rows.append(unsloth_layer_row)

    swiglu_row, swiglu_out = _execute_entry("liger_swiglu", seed + 4)
    rows.append(swiglu_row)
    softmax_row, _ = _execute_entry("liger_softmax", seed + 5, {"x": swiglu_out})
    rows.append(softmax_row)
    for offset, kernel_id in enumerate(("liger_swiglu_tiled", "liger_geglu", "liger_fused_swiglu"), start=6):
        row, _ = _execute_entry(kernel_id, seed + offset)
        rows.append(row)
    passed = all(row["path"] == "Fast" and all(check["correct"] for check in row["checks"]) for row in rows)
    return {
        "workload": workload.id,
        "status": "passed" if passed else "failed",
        "execution": "固定模型层数据流加官方算子变体；随机参数，无预训练权重",
        "steps": rows,
        "kernel_ids": [row["kernel_id"] for row in rows],
        "evidence": "真实 wrapper 语义与模型层形状的冻结轨迹回放；不代表完整模型训练吞吐。",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("workload", choices=("inductor_custom_kernel_graph", "transformer_layer_trace"))
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    result = run_inductor_workload(args.seed) if args.workload == "inductor_custom_kernel_graph" else run_transformer_workload(args.seed)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())

