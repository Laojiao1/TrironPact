"""隔离核对 e2 Guard 放行、原始 Kernel 启动与独立参考。"""

from __future__ import annotations

import argparse
import json

import torch

from bench.e1.catalog import load
from bench.e1.specs import construct, reference, value
from bench.e1.worker import decode
from bench.e2.coverage import _kernel
from bench.e2.plans import build_plan


def execute(kernel_id: str) -> dict:
    entry = next(item for item in load()["entries"] if item["id"] == kernel_id)
    if entry["split"] not in ("development", "holdout") or not entry["smoke"]:
        raise ValueError("Unsupported：e2 Fast 核对只接受正式正向集")
    spec = decode(entry["smoke"])
    kernel = _kernel(entry)
    plan = build_plan(entry)
    if plan.status != "Supported":
        return {"id": kernel_id, "status": plan.status, "reason": plan.reason, "plan_fingerprint": plan.fingerprint, "runs": []}
    runs = []
    for seed in (17, 42):
        tensors = construct(spec, seed)
        guard = plan.evaluate(tensors)
        if not guard.allowed:
            runs.append({"seed": seed, "guard": {"status": guard.status, "reason": guard.reason, "trace": guard.trace}, "checks": []})
            continue
        expected = reference(spec, tensors)
        arguments = {binding.parameter: value(binding.expression, tensors) for binding in spec.bindings}
        if tuple(arguments) != tuple(kernel.arg_names):
            raise ValueError("Unknown：JIT 参数与冻结 wrapper 绑定不一致")
        kernel[spec.grid](**arguments, num_warps=4)
        torch.cuda.synchronize()
        checks = []
        for name, wanted in zip(spec.outputs, expected):
            actual = tensors[name]
            compatible = actual.shape == wanted.shape and actual.dtype == wanted.dtype
            close = torch.isclose(actual, wanted, rtol=spec.rtol, atol=spec.atol, equal_nan=False) if compatible else None
            correct = compatible and bool(close.all().item())
            checks.append({"output": name, "correct": correct, "mismatch_count": int((~close).sum().item()) if close is not None else None})
        runs.append(
            {
                "seed": seed,
                "guard": {"status": guard.status, "reason": guard.reason, "trace": guard.trace},
                "checks": checks,
                "tensor_metadata": {
                    name: {
                        "shape": list(tensor.shape),
                        "stride_elements": list(tensor.stride()),
                        "storage_offset_elements": tensor.storage_offset(),
                        "dtype": str(tensor.dtype),
                    }
                    for name, tensor in tensors.items()
                },
            }
        )
    passed = all(run["guard"]["status"] == "True" and run["checks"] and all(check["correct"] for check in run["checks"]) for run in runs)
    return {
        "id": kernel_id,
        "status": "passed" if passed else "failed",
        "plan_fingerprint": plan.fingerprint,
        "runs": runs,
        "evidence": "冻结调用域内 Guard+原始 Kernel+独立参考的有界经验核对；不证明一般数值语义",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("id")
    args = parser.parse_args()
    result = execute(args.id)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
