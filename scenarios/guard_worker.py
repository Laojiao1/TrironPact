"""第五阶段隔离数值 Oracle：单进程只运行一个指定输入和路径。"""

from __future__ import annotations

import json
import sys

import torch

from pact.dispatch import dispatch
from scenarios.cases import make_case, reference


def make_input(recipe: dict):
    name, layout = recipe["name"], recipe["layout"]
    if name in ("A", "A2", "B", "D"):
        return make_case(name, layout, m=recipe.get("m", 7), n=recipe.get("n", 11))
    if name == "holdout_vector":
        n = recipe.get("n", 129)
        return torch.arange(n * 2, device="cuda", dtype=torch.float32)[::2]
    if name == "holdout_matrix":
        m, n = recipe.get("m", 7), recipe.get("n", 11)
        return torch.arange(m * (2 * n + 3), device="cuda", dtype=torch.float32).reshape(m, 2 * n + 3)[:, :2 * n:2]
    if name in ("pointer_hint", "index_hint"):
        n = recipe.get("n", 128)
        source = torch.arange(n + (1 if layout == "offset" else 0), device="cuda", dtype=torch.float32)
        return source[1:] if layout == "offset" else source
    raise ValueError("未知隔离配方")


def run(recipe: dict) -> dict:
    value = make_input(recipe)
    x, y = value if isinstance(value, tuple) else (value, None)
    try:
        result = dispatch(recipe["name"], x, y, repair_policy=recipe.get("policy", "cost"))
        if result.output is None:
            return {"classification": "Unsupported/Unknown", "detail": result.to_dict()}
        expected = reference(x, y)
        torch.cuda.synchronize()
        equal = bool(torch.allclose(result.output, expected, rtol=1e-5, atol=1e-5))
        return {"classification": "correct" if equal else "wrong_value", "detail": result.to_dict(),
                "shape": tuple(x.shape), "stride": tuple(x.stride()), "storage_offset": x.storage_offset(),
                "effective_ptr_mod16": x.data_ptr() % 16, "equal": equal}
    except Exception as error:
        return {"classification": "exception", "error_type": type(error).__name__, "message": str(error)}


if __name__ == "__main__":
    print(json.dumps(run(json.loads(sys.stdin.read())), ensure_ascii=False))
