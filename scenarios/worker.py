"""本文件在独立进程中运行单个 GPU 用例，隔离子进程入口，执行并输出首错位置、数值比对等明细，避免错误污染主进程的 CUDA 状态。"""

import argparse
import json
import traceback

import torch

from scenarios.cases import make_case, reference
from scenarios.kernels import run_fast
from pact.runtime import dispatch


def run(name: str, layout: str, m: int, n: int, mode: str, refined: bool = False) -> dict:
    inputs = make_case(name, layout, m, n)
    x, y = inputs if isinstance(inputs, tuple) else (inputs, None)
    input_ptr = x.data_ptr()
    materialized = x.contiguous()
    contiguous_copied = materialized.data_ptr() != input_ptr
    expected = reference(x, y, case=name)
    if mode == "raw":
        actual = run_fast(name, x, y)
        path, decision = "Raw Fast", None
    else:
        actual, path, decision = dispatch(name, x, y, refined=refined)
    torch.cuda.synchronize()

    result = {
        "case": name,
        "layout": layout,
        "mode": mode,
        "refined_contract": refined,
        "shape": list(x.shape),
        "stride": list(x.stride()),
        "second_stride": None if y is None else list(y.stride()),
        "storage_offset": x.storage_offset(),
        "effective_ptr_mod_16": x.data_ptr() % 16,
        "input_ptr_unchanged": x.data_ptr() == input_ptr,
        "is_contiguous": x.is_contiguous(),
        "contiguous_copied": contiguous_copied,
        "path": path,
        "guard": None if decision is None else {
            "allowed": decision.allowed,
            "reason": decision.reason,
            "evidence": decision.evidence,
            "proof_scope": decision.proof_scope,
            "checks": list(decision.checks),
        },
        "expected_shape": list(expected.shape),
        "actual_shape": list(actual.shape),
        "expected_dtype": str(expected.dtype),
        "actual_dtype": str(actual.dtype),
    }
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        result["kind"] = "metadata_mismatch"
        return result
    rtol, atol = 0.0, 0.0
    result["tolerance"] = {"rtol": rtol, "atol": atol}
    result["max_abs_error"] = float((actual.float() - expected.float()).abs().max().item())
    matches = torch.isclose(actual, expected, rtol=rtol, atol=atol, equal_nan=True)
    result["correct"] = bool(matches.all().item())
    if not result["correct"]:
        first = (~matches).nonzero()[0].tolist()
        result["first_mismatch"] = first
        result["expected_value"] = expected[tuple(first)].item()
        result["actual_value"] = actual[tuple(first)].item()
        result["mismatch_count"] = int((~matches).sum().item())
        result["kind"] = "numeric_mismatch"
    else:
        result["kind"] = "correct"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="隔离运行一个 TritonPact 用例")
    parser.add_argument("name", choices=("A", "A2", "B", "D"))
    parser.add_argument("layout")
    parser.add_argument("--m", type=int, default=7)
    parser.add_argument("--n", type=int, default=11)
    parser.add_argument("--mode", choices=("raw", "dispatch"), default="dispatch")
    parser.add_argument("--refined", action="store_true", help="启用已修订的单维条件式")
    args = parser.parse_args()
    try:
        result = run(args.name, args.layout, args.m, args.n, args.mode, args.refined)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except Exception as error:
        # 子进程仍可能已损坏 CUDA Context；主进程只读取错误记录。
        print(json.dumps({"kind": "exception", "type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
