"""隔离核对 e4a 中安全但超出 e2 冻结域的真实上游布局。"""

from __future__ import annotations

import json

import torch

from bench.e1.catalog import load
from bench.e1.specs import construct, reference
from bench.e1.worker import decode
from bench.e2.plans import build_plan
from bench.e3.worker import _checks, _launch


def run_liger_softmax_row_padding(seed: int = 17) -> dict:
    """行 stride 被 Kernel 显式传入，但 e2 冻结元数据域仍保守拒绝。

    该运行只证明这个固定 padding 输入数值正确，不能绕过 Guard 授予 Fast。
    上游 wrapper 的 `.contiguous()` 会为此输入实际分配并复制。
    """

    entry = next(item for item in load()["entries"] if item["id"] == "liger_softmax")
    spec = decode(entry["smoke"])
    tensors = construct(spec, seed)
    original = tensors["x"]
    base = torch.empty((original.shape[0], original.shape[1] + 13), device=original.device, dtype=original.dtype)
    padded = base[:, : original.shape[1]]
    padded.copy_(original)
    tensors["x"] = padded
    guard = build_plan(entry).evaluate(tensors)
    expected = reference(spec, tensors)
    _launch(entry, tensors)
    contiguous = padded.contiguous()
    checks = _checks(entry, tensors, expected)
    return {
        "case": "liger_softmax_row_padding",
        "kernel_id": "liger_softmax",
        "classification": "correct_outside_frozen_guard_domain" if all(item["correct"] for item in checks) else "numeric_mismatch",
        "guard": {"status": guard.status, "allowed": guard.allowed, "reason": guard.reason, "trace": guard.trace},
        "checks": checks,
        "layout": {
            "shape": list(padded.shape),
            "stride_elements": list(padded.stride()),
            "storage_offset_elements": padded.storage_offset(),
            "is_contiguous": padded.is_contiguous(),
        },
        "upstream_contiguous_copy_bytes": padded.numel() * padded.element_size() if contiguous.data_ptr() != padded.data_ptr() else 0,
        "scope": "e2 Guard 的声明域冻结 exact shape/stride；本例是能力缺口，不是可在线放行路径。",
    }


def main() -> int:
    result = run_liger_softmax_row_padding()
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return 0 if result["classification"] == "correct_outside_frozen_guard_domain" and not result["guard"]["allowed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

