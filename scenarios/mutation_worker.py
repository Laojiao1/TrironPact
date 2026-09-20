"""单次变异的独立 GPU 子进程；stdin/stdout 为受限 JSON 协议。"""

from __future__ import annotations

import json
import sys
import traceback

import torch
import triton

from pact.mutation import MutationRecipe
from scenarios.alignment_fixtures import run_index_hint, run_pointer_hint
from scenarios.cases import reference
from scenarios.holdouts import run_strided_vector, run_two_stride_matrix
from scenarios.kernels import run_fast


def _tensor(spec):
    storage = torch.arange(spec.storage_size, device="cuda", dtype=torch.float32)
    return torch.as_strided(storage, spec.shape, spec.stride, spec.offset)


def execute(recipe: MutationRecipe) -> dict:
    torch.manual_seed(recipe.seed)
    tensors = {name: _tensor(spec) for name, spec in recipe.inputs}
    x, y = tensors["X"], tensors.get("Y")
    meta = {name: {"shape": list(tensor.shape), "stride": list(tensor.stride()), "storage_offset": tensor.storage_offset(), "effective_ptr": tensor.data_ptr(), "effective_ptr_mod_16": tensor.data_ptr() % 16, "storage_nbytes": tensor.untyped_storage().nbytes()} for name, tensor in tensors.items()}
    environment = {"torch": torch.__version__, "triton": triton.__version__, "cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(x.device)}
    if recipe.case == "pointer" and x.data_ptr() % 16 != 0:
        return {"id": recipe.id, "fingerprint": recipe.fingerprint, "case": recipe.case, "label": recipe.label, "kind": "preflight_skip", "reason": "实际有效指针不满足 tl.multiple_of 优化前提", "input_metadata": meta, "environment": environment}
    expected = reference(x, y, case=recipe.case)
    torch.cuda.synchronize()
    runners = {"holdout_vector": run_strided_vector, "holdout_matrix": run_two_stride_matrix, "pointer": run_pointer_hint, "index": run_index_hint}
    actual = runners[recipe.case](x) if recipe.case in runners else run_fast(recipe.case, x, y)
    torch.cuda.synchronize()
    result = {"id": recipe.id, "fingerprint": recipe.fingerprint, "case": recipe.case, "label": recipe.label, "input_metadata": meta, "environment": environment, "reference_symbol": "scenarios.cases.reference", "expected_shape": list(expected.shape), "actual_shape": list(actual.shape), "expected_dtype": str(expected.dtype), "actual_dtype": str(actual.dtype), "tolerance": {"rtol": 0.0, "atol": 0.0}}
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        return {**result, "kind": "metadata_mismatch"}
    equal = torch.isclose(expected, actual, rtol=0.0, atol=0.0, equal_nan=True)
    torch.cuda.synchronize()
    if bool(equal.all().item()):
        return {**result, "kind": "correct", "mismatch_count": 0}
    first = (~equal).nonzero()[0].tolist()
    return {**result, "kind": "numeric_mismatch", "first_mismatch": first, "expected_value": expected[tuple(first)].item(), "actual_value": actual[tuple(first)].item(), "mismatch_count": int((~equal).sum().item())}


def main() -> int:
    identity = {}
    stage = "protocol"
    try:
        raw = sys.stdin.read(65537)
        if len(raw) > 65536:
            raise ValueError("JSON 配方超过字节预算")
        recipe = MutationRecipe.from_dict(json.loads(raw))
        identity = {"id": recipe.id, "fingerprint": recipe.fingerprint, "case": recipe.case, "label": recipe.label}
        stage = "execution"
        result = execute(recipe)
        code = 0
    except Exception as exc:
        result = {**identity, "kind": "worker_exception", "stage": stage, "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc(limit=8)[-3000:]}
        code = 2
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
