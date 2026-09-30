"""逐 Kernel 独立进程核对冻结参考和参数绑定，不接入在线分派。"""

from __future__ import annotations

import argparse
import importlib.util
import json
import torch

from bench.e1.acquire import ROOT
from bench.e1.catalog import load
from bench.e1.specs import Binding, SmokeSpec, TensorSpec, construct, reference, value


def decode(raw: dict) -> SmokeSpec:
    """仅反序列化 schema 内字段，缺失字段/无效绑定不能隐式填真值。"""
    return SmokeSpec(tuple(TensorSpec(t["name"], tuple(t["shape"]), t["dtype"], t["fill"]) for t in raw["tensors"]),
                     tuple(raw["inputs"]), tuple(raw["outputs"]), tuple(Binding(**b) for b in raw["bindings"]),
                     tuple(raw["grid"]), raw["reference"], raw["rtol"], raw["atol"])


def execute(id: str) -> dict:
    """只执行非空、连续、新分配且无 alias 的登记 smoke 输入。

    以原始 JIT 签名逐参数启动，同时核对全部输出和辅助缓存。
    编译/运行异常由父进程分类为 Unknown 并保留 stderr；数值不一致
    记录 failed，不因此调宽容差或修改 Kernel。返回结果不授予 Fast。
    """
    row = next(r for r in load()["entries"] if r["id"] == id)
    if row["smoke"] is None:
        raise ValueError("Unsupported：挑战样本不执行数值 smoke")
    spec = decode(row["smoke"])
    path = ROOT / row["kernel_module"]
    module_spec = importlib.util.spec_from_file_location(f"e1_kernel_{id}", path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    kernel = getattr(module, row["function"].split(".")[-1])
    results = []
    for seed in (17, 42):
        tensors = construct(spec, seed)
        storages = [t.untyped_storage().data_ptr() for t in tensors.values()]
        if len(storages) != len(set(storages)):
            raise ValueError("Unknown：smoke 存在 storage alias")
        expected = reference(spec, tensors)
        arguments = {b.parameter: value(b.expression, tensors) for b in spec.bindings}
        # 所有指针直接取实际 Tensor，stride 是元素单位，不再添加 view offset。
        if tuple(arguments) != tuple(kernel.arg_names):
            raise ValueError("Unknown：实际 JIT 参数与 wrapper 规格不一致")
        kernel[spec.grid](**arguments, num_warps=4)
        torch.cuda.synchronize()
        checks = []
        for name, ref in zip(spec.outputs, expected):
            actual = tensors[name]
            compatible = actual.shape == ref.shape and actual.dtype == ref.dtype
            close = torch.isclose(actual, ref, rtol=spec.rtol, atol=spec.atol, equal_nan=False) if compatible else None
            matches = compatible and bool(close.all().item())
            checks.append({"output": name, "correct": matches, "mismatch_count": int((~close).sum().item()) if close is not None else None})
        results.append({"seed": seed, "checks": checks, "parameters": {b.parameter: b.expression for b in spec.bindings},
                        "tensor_metadata": {n: {"shape": list(t.shape), "stride_elements": list(t.stride()), "storage_offset_elements": t.storage_offset(), "dtype": str(t.dtype)} for n,t in tensors.items()}})
    return {"id": id, "status": "passed" if all(c["correct"] for r in results for c in r["checks"]) else "failed",
            "runs": results, "evidence": "Empirically-Validated：仅两个冻结输入，不证明一般安全，不授予 Fast"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("id")
    args = parser.parse_args()
    result = execute(args.id)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
