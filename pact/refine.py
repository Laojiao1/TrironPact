"""边界见证（Witnesses）驱动的谓词修订与静态推导证明"""

from dataclasses import asdict, replace

from pact.analysis import Analysis


def refine_singleton_strides(contract: Analysis, witnesses: list[dict]) -> tuple[Analysis, dict]:
    """仅处理二维线性读取：只有一行或一列时，对应 stride 不参与地址。"""
    if contract.name != "A" or contract.status != "Supported":
        return contract, {"changed": False, "reason": "不属于可修订的二维线性模板"}

    accepted: dict[int, dict] = {}
    for record in witnesses:
        detail = record.get("detail", {})
        if record.get("category") != "correct" or detail.get("case") != "A" or detail.get("mode") != "raw":
            continue
        shape, stride = detail.get("shape"), detail.get("stride")
        if not shape or not stride or len(shape) != 2 or len(stride) != 2:
            continue
        if shape[0] == 1 and shape[1] > 1 and stride[0] != shape[1] and stride[1] == 1:
            accepted[0] = record
        if shape[1] == 1 and shape[0] > 1 and stride[0] == shape[1] and stride[1] != 1:
            accepted[1] = record

    if set(accepted) != {0, 1}:
        return contract, {"changed": False, "reason": "尚未取得分别违反两条候选谓词的正确边界输入", "accepted_dimensions": sorted(accepted)}

    # 反例只说明原条件过强；条件式还要由行列索引的取值范围证明。
    revised = tuple(replace(predicate, unless_size_one=predicate.dimension) for predicate in contract.predicates)
    result = replace(contract, predicates=revised, refinement_state="两个单谓词补集输入触发修订；条件式由有限索引范围推导")
    evidence = {
        "changed": True,
        "before": [asdict(item) for item in contract.predicates],
        "after": [asdict(item) for item in revised],
        "witnesses": [{"shape": accepted[axis]["detail"]["shape"], "stride": accepted[axis]["detail"]["stride"], "violated_predicate_dimension": axis} for axis in (0, 1)],
        "static_reason": "二维线性 idx=row*N+col；M=1 时 row 恒为 0，N=1 时 col 恒为 0，因此对应 stride 不影响任何读取地址。",
        "scope": "仅适用于 A 的二维线性读取模板和声明的逐元素语义",
    }
    return result, evidence


def apply_proven_singleton_rule(contract: Analysis) -> Analysis:
    """运行时使用已审定的条件式；只对 A 的固定模板生效。"""
    if contract.name != "A" or contract.status != "Supported":
        return contract
    return replace(contract, predicates=tuple(replace(item, unless_size_one=item.dimension) for item in contract.predicates), refinement_state="单维尺寸为 1 时跳过无作用的 stride")
