"""第五阶段离线成本表；在线仅做有域检查的确定性查表。"""

from __future__ import annotations

import hashlib
import json
import math
import platform
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

import torch
import triton

from pact.guard_plan import GuardPlan


COST_VERSION = "cost-table-2"


@lru_cache(maxsize=1)
def repair_fingerprint() -> str:
    source = Path(__file__).with_name("relayout.py").read_bytes()
    return hashlib.sha256(source).hexdigest()


def shape_bucket(shape: tuple[int, ...]) -> str:
    # 维度桶和 Tile 尾部边界都进入主键，避免 B-1/B/B+1 混用。
    bands = tuple("1" if n == 1 else "2-8" if n <= 8 else "9-16" if n <= 16 else
                  "17-127" if n < 128 else "128" if n == 128 else "129-1024" if n <= 1024 else ">1024" for n in shape)
    return "x".join(bands) + (":tail" if shape[-1] % 128 else ":full")


def layout_family(tensors: dict[str, torch.Tensor]) -> str:
    def one(t: torch.Tensor) -> str:
        if t.is_contiguous():
            return "offset" if t.storage_offset() else "contiguous"
        if t.ndim == 2 and t.stride(0) == 1:
            return "transposed"
        if t.ndim == 2 and t.stride(1) == 1:
            return "row_padded"
        return "strided"
    return "+".join(f"{name}:{one(t)}" for name, t in sorted(tensors.items()))


@dataclass(frozen=True)
class CostEntry:
    key: tuple[str, str, str]
    fingerprint: str
    dtype: str
    device: str
    copied: tuple[str, ...]
    alignment: tuple[tuple[str, int], ...]
    exact_shape: tuple[int, ...]
    median_us: tuple[tuple[str, float], ...]
    samples: int
    stable: bool = True

    def __post_init__(self) -> None:
        if (len(self.key) != 3 or any(not isinstance(part, str) or not part for part in self.key)
                or not isinstance(self.fingerprint, str) or len(self.fingerprint) != 64
                or not isinstance(self.dtype, str) or not isinstance(self.device, str)
                or any(not isinstance(name, str) for name in self.copied)
                or any(not isinstance(name, str) or type(rem) is not int or not 0 <= rem < 16 for name, rem in self.alignment)
                or not self.exact_shape or any(type(dim) is not int or dim < 1 for dim in self.exact_shape)
                or not self.median_us or any(not isinstance(path, str) or type(value) not in (int, float)
                                             or not math.isfinite(value) or value < 0 for path, value in self.median_us)
                or type(self.samples) is not int or self.samples < 1 or type(self.stable) is not bool):
            raise ValueError("成本表条目字段无效")

    def to_dict(self) -> dict:
        return asdict(self)


class CostTable:
    def __init__(self, entries: tuple[CostEntry, ...] = (), version: str = COST_VERSION,
                 repair_digest: str | None = None):
        self.entries = entries
        self.version = version
        self.repair_digest = repair_digest or repair_fingerprint()

    @classmethod
    def from_report(cls, path: str | Path) -> CostTable:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
            if data.get("version") != COST_VERSION or data.get("repair_fingerprint") != repair_fingerprint() or not torch.cuda.is_available():
                return cls(version="stale")
            env = data.get("environment", {})
            if env.get("python") != platform.python_version() or env.get("torch") != torch.__version__ or env.get("triton") != triton.__version__ or env.get("cuda") != torch.version.cuda or env.get("gpu") != torch.cuda.get_device_name():
                return cls(version="stale")
            entries = tuple(CostEntry(tuple(item["key"]), item["fingerprint"], item["dtype"], item["device"],
                                      tuple(item["copied"]), tuple(tuple(pair) for pair in item["alignment"]),
                                      tuple(item["exact_shape"]), tuple(tuple(pair) for pair in item["median_us"]),
                                      item["samples"], item.get("stable", False)) for item in data["table"])
            return cls(entries, repair_digest=data["repair_fingerprint"])
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            return cls(version="stale")

    def has_repair_entry(self, plan: GuardPlan, tensors: dict[str, torch.Tensor]) -> bool:
        if self.version != COST_VERSION or self.repair_digest != repair_fingerprint() or not tensors or not self.entries:
            return False
        first = next(iter(tensors.values()))
        shape = tuple(first.shape)
        key = (plan.name, shape_bucket(shape), layout_family(tensors))
        alignment = tuple((name, tensor.data_ptr() % 16) for name, tensor in sorted(tensors.items()))
        return any(item.key == key and item.fingerprint == plan.fingerprint and item.dtype == str(first.dtype)
                   and item.device == str(first.device) and item.exact_shape == shape and item.alignment == alignment
                   and item.copied and item.samples > 0 and item.stable and any(path == "Relayout+Fast" for path, _ in item.median_us)
                   for item in self.entries)

    def select(self, plan: GuardPlan, tensors: dict[str, torch.Tensor], feasible: set[str],
               copied: tuple[str, ...] = ()) -> tuple[str, str]:
        default = "Fast" if "Fast" in feasible else "PyTorch Fallback" if "PyTorch Fallback" in feasible else "Unsupported/Unknown"
        if self.version != COST_VERSION or self.repair_digest != repair_fingerprint() or not tensors or not feasible or not self.entries:
            return default, "成本表缺失或版本不符，使用确定性默认"
        first = next(iter(tensors.values()))
        shape = tuple(first.shape)
        key = (plan.name, shape_bucket(shape), layout_family(tensors))
        alignment = tuple((name, tensor.data_ptr() % 16) for name, tensor in sorted(tensors.items()))
        matches = [item for item in self.entries if item.key == key and item.fingerprint == plan.fingerprint
                   and item.dtype == str(first.dtype) and item.device == str(first.device)
                   and item.copied == copied and item.alignment == alignment and item.exact_shape == shape
                   and item.samples > 0 and item.stable]
        if len(matches) != 1:
            return default, "无唯一且完整匹配的离线成本条目"
        costs = {path: value for path, value in matches[0].median_us if path in feasible and value >= 0}
        if set(costs) != feasible:
            return default, "成本条目未覆盖全部安全可行路径"
        return min(sorted(costs), key=costs.__getitem__), "匹配离线成本条目"
