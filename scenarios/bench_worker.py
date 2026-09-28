"""第六阶段离线基准的独立 GPU worker；每次只执行一个 Kernel 的边界矩阵。"""

from __future__ import annotations

import json
import sys
import traceback

import torch
import triton

from bench.specs import NEW_CASES


def _input(layout: str, n: int, shift: float) -> torch.Tensor:
    storage = torch.arange(2 * n + 4, device="cuda", dtype=torch.float32) / 17.0 + shift
    if layout == "contiguous":
        return storage[:n].clone()
    if layout == "offset":
        return storage[1:n + 1]
    if layout == "strided":
        return storage[:2 * n:2]
    raise ValueError(f"未知布局：{layout}")


def _meta(tensor: torch.Tensor) -> dict:
    return {"shape": list(tensor.shape), "stride": list(tensor.stride()), "storage_offset": tensor.storage_offset()}


def _feature_inputs(layout: str, m: int = 7, n: int = 11) -> tuple[torch.Tensor, torch.Tensor]:
    x_storage = torch.arange(2 * m * n + 8, device="cuda", dtype=torch.float32) / 19.0 - 2.0
    f_storage = torch.arange(2 * n + 4, device="cuda", dtype=torch.float32) / 13.0 + 0.5
    if layout == "contiguous":
        return x_storage[:m * n].reshape(m, n).clone(), f_storage[:n].clone()
    if layout == "offset":
        return torch.as_strided(x_storage, (m, n), (n, 1), 1), f_storage[1:n + 1]
    if layout == "x_col_strided":
        return torch.as_strided(x_storage, (m, n), (2 * n, 2), 0), f_storage[:n].clone()
    if layout == "feature_strided":
        return x_storage[:m * n].reshape(m, n).clone(), f_storage[:2 * n:2]
    raise ValueError(f"未知特征广播布局：{layout}")


def execute(name: str) -> dict:
    wrapper, reference, *_ = NEW_CASES[name]
    feature = name.startswith("bench_feature_")
    layouts = ("contiguous", "offset", "x_col_strided", "feature_strided") if feature else ("contiguous", "offset", "strided")
    rows = []
    for layout in layouts:
        if feature:
            x, second = _feature_inputs(layout)
        else:
            x = _input(layout, 129, -4.0)
            second = _input(layout, 129, 2.0) if name in {"bench_multiply", "bench_axpy"} else None
        args = (x, second) if second is not None else (x,)
        expected = reference(*args)
        torch.cuda.synchronize()
        actual = wrapper(*args)
        torch.cuda.synchronize()
        close = torch.isclose(actual, expected, rtol=1e-6, atol=1e-6, equal_nan=True)
        torch.cuda.synchronize()
        mismatch = int((~close).sum().item())
        if feature:
            eligible = layout in {"contiguous", "offset"}
        else:
            eligible = layout != "strided"
        rows.append({
            "layout": layout,
            "truth": "eligible" if eligible else "ineligible",
            "observation": "correct" if mismatch == 0 else "numeric_mismatch",
            "mismatch_count": mismatch,
            "inputs": {"X": _meta(x), **({("F" if feature else "Z"): _meta(second)} if second is not None else {})},
        })
    return {"kernel": name, "rows": rows, "environment": {"torch": torch.__version__, "triton": triton.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0)}}


def main() -> int:
    try:
        result = execute(sys.argv[1])
        code = 0
    except Exception as error:
        result = {"kernel": sys.argv[1] if len(sys.argv) > 1 else "", "kind": "worker_exception", "type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc(limit=8)[-3000:]}
        code = 2
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
