"""第四阶段：有界、可复现的物理布局变异配方与候选影子评价。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

from pact.candidates import Candidate
from pact.contract_dsl import TensorMetadata


CASES = {"A", "A2", "B", "D", "holdout_vector", "holdout_matrix", "pointer", "index"}
MAX_ELEMENTS = 8192


@dataclass(frozen=True)
class TensorRecipe:
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    offset: int = 0
    storage_elements: int | None = None

    def __post_init__(self) -> None:
        if len(self.shape) not in (1, 2) or len(self.shape) != len(self.stride):
            raise ValueError("只支持一维/二维 Tensor 配方")
        if any(type(x) is not int or x < 1 or x > MAX_ELEMENTS for x in self.shape):
            raise ValueError("尺寸超出非空有界支持域")
        if any(type(x) is not int or x < 0 or x > MAX_ELEMENTS for x in self.stride) or type(self.offset) is not int or self.offset < 0:
            raise ValueError("步长或 offset 无效")
        need = self.offset + 1 + sum((n - 1) * step for n, step in zip(self.shape, self.stride))
        storage = need if self.storage_elements is None else self.storage_elements
        if type(storage) is not int or storage < need or storage > MAX_ELEMENTS:
            raise ValueError("配方不能构造合法的有界 PyTorch 视图")
        if self.numel > MAX_ELEMENTS:
            raise ValueError("Tensor 超出元素预算")

    @property
    def numel(self) -> int:
        value = 1
        for item in self.shape:
            value *= item
        return value

    @property
    def storage_size(self) -> int:
        return self.storage_elements if self.storage_elements is not None else self.offset + 1 + sum((n - 1) * step for n, step in zip(self.shape, self.stride))

    @classmethod
    def from_dict(cls, data: dict) -> TensorRecipe:
        if not isinstance(data, dict) or set(data) != {"shape", "stride", "offset", "storage_elements"}:
            raise ValueError("Tensor 配方字段缺失或未知")
        return cls(tuple(data["shape"]), tuple(data["stride"]), data["offset"], data["storage_elements"])


@dataclass(frozen=True)
class MutationRecipe:
    version: int
    case: str
    label: str
    inputs: tuple[tuple[str, TensorRecipe], ...]
    target_family: str
    seed: int = 0

    def __post_init__(self) -> None:
        if self.version != 1 or self.case not in CASES or not self.label or self.target_family not in {"shape_stride", "tile_mask", "alignment", "span"} or type(self.seed) is not int:
            raise ValueError("变异配方版本、案例或目标无效")
        names = tuple(key for key, _ in self.inputs)
        if names != (("X", "Y") if self.case == "D" else ("X",)):
            raise ValueError("输入身份或顺序无效")
        shapes = {value.shape for _, value in self.inputs}
        if len(shapes) != 1 or (self.case in {"A", "A2", "holdout_matrix"}) != (len(next(iter(shapes))) == 2):
            raise ValueError("输入维度与案例不匹配")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> MutationRecipe:
        if not isinstance(data, dict) or set(data) != {"version", "case", "label", "inputs", "target_family", "seed"}:
            raise ValueError("变异配方字段缺失或未知")
        return cls(data["version"], data["case"], data["label"], tuple((key, TensorRecipe.from_dict(spec)) for key, spec in data["inputs"]), data["target_family"], data["seed"])

    @property
    def fingerprint(self) -> str:
        raw = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    @property
    def id(self) -> str:
        return f"{self.case}:{self.label}:{self.fingerprint[:12]}"


def _one(case: str, label: str, shape: tuple[int, ...], stride: tuple[int, ...], offset: int, family: str) -> MutationRecipe:
    return MutationRecipe(1, case, label, (("X", TensorRecipe(shape, stride, offset)),), family)


def standard_recipes() -> tuple[MutationRecipe, ...]:
    """固定种子和预算；名称只用于构造样例，不参与候选提取规则。"""
    recipes = []
    for case in ("A", "A2", "holdout_matrix"):
        recipes.extend((
            _one(case, "contiguous", (7, 11), (11, 1), 0, "shape_stride"),
            _one(case, "padded", (7, 11), (14, 1), 0, "shape_stride"),
            _one(case, "transpose", (7, 11), (1, 7), 0, "shape_stride"),
            _one(case, "single_row", (1, 11), (14, 1), 0, "shape_stride"),
            _one(case, "single_col", (7, 1), (1, 3), 0, "shape_stride"),
        ))
    for case in ("B", "holdout_vector", "pointer", "index"):
        recipes.extend((
            _one(case, "tile_minus", (127,), (1,), 0, "tile_mask"),
            _one(case, "tile_exact", (128,), (1,), 0, "tile_mask"),
            _one(case, "tile_plus", (129,), (1,), 0, "tile_mask"),
            _one(case, "offset", (129,), (1,), 1, "alignment" if case in ("pointer", "index") else "span"),
        ))
        if case in ("B", "holdout_vector"):
            recipes.append(_one(case, "stride_two", (129,), (2,), 0, "shape_stride"))
    for label, xstep, ystep in (("contiguous", 1, 1), ("x_stride", 2, 1), ("y_stride", 1, 2)):
        recipes.append(MutationRecipe(1, "D", label, (("X", TensorRecipe((129,), (xstep,))), ("Y", TensorRecipe((129,), (ystep,)))), "shape_stride"))
    return tuple(recipes)


def symbolic_metadata(recipe: MutationRecipe, *, ptrs: dict[str, int] | None = None) -> tuple[dict[str, TensorMetadata], dict[str, int]]:
    ptrs = ptrs or {}
    tensors = {name: TensorMetadata(spec.shape, spec.stride, spec.offset, ptrs.get(name, 0), 4, 4 * spec.storage_size) for name, spec in recipe.inputs}
    shape = recipe.inputs[0][1].shape
    out_stride = (1,) if len(shape) == 1 else (shape[1], 1)
    out_count = 1
    for dim in shape:
        out_count *= dim
    tensors["OUT"] = TensorMetadata(shape, out_stride, 0, ptrs.get("OUT", 0), 4, 4 * out_count)
    if len(shape) == 1:
        n = shape[0]
        scalars = {"N": n, "n_elements": n, "B": 128, "BLOCK_SIZE": 128, "S": recipe.inputs[0][1].stride[0]}
    else:
        m, n = shape
        scalars = {"M": m, "N": n, "B": 128 if recipe.case == "A" else 1 << (n - 1).bit_length(), "S0": recipe.inputs[0][1].stride[0], "S1": recipe.inputs[0][1].stride[1]}
    return tensors, scalars


def shadow_candidates(recipe: MutationRecipe, candidates: list[dict], *, ptrs: dict[str, int] | None = None) -> list[dict]:
    tensors, scalars = symbolic_metadata(recipe, ptrs=ptrs)
    return [{"candidate": index, "purpose": data["purpose"], "evidence": data["evidence"], "origin": data["origin"], "rule": data["rule"], "value": Candidate.from_dict(data).evaluate(tensors, scalars)} for index, data in enumerate(candidates)]
