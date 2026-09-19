"""保存外部给定的算子语义、参数映射和参考实现入口，供访存分析追溯。"""

from dataclasses import dataclass


@dataclass(frozen=True)
class InputMeaning:
    tensor: str
    pointer: str
    shape_params: tuple[str, ...]
    logical_read: str


@dataclass(frozen=True)
class OperatorMeaning:
    name: str
    axes: tuple[str, ...]
    index_domain: str
    inputs: tuple[InputMeaning, ...]
    output_pointer: str
    logical_write: str
    launch_bindings: tuple[tuple[str, str], ...]
    reference_symbol: str
    reference_rule: str
    assumptions: tuple[str, ...]

    @property
    def ndim(self) -> int:
        return len(self.axes)

    def input_for(self, tensor: str) -> InputMeaning:
        return next(item for item in self.inputs if item.tensor == tensor)


# 这是算子调用方声明的语义锚点；这里不预填待恢复的 stride 条件。
# launch_bindings 描述当前 wrapper 如何传参，属于分析前提，不声称由 Kernel AST 自动恢复。
SEMANTICS = {
    "A": OperatorMeaning(
        "A", ("row", "col"), "0 <= row < M，0 <= col < N",
        (InputMeaning("X", "X", ("M", "N"), "x[row, col]"),),
        "Y", "out[row, col] = x[row, col] + 1",
        (("M", "x.size(0)"), ("N", "x.size(1)"), ("B", "128"), ("grid", "ceil(M*N/B)")),
        "scenarios.cases.reference", "reference(x) = x + 1",
        ("CUDA float32", "各维长度至少为 1", "元素总数不超过 1000000", "显式尾部 mask"),
    ),
    "A2": OperatorMeaning(
        "A2", ("row", "col"), "0 <= row < M，0 <= col < N",
        (InputMeaning("X", "X", ("M", "N"), "x[row, col]"),),
        "Y", "out[row, col] = x[row, col] + 1",
        (("M", "x.size(0)"), ("N", "x.size(1)"), ("S0", "x.stride(0)"), ("B", "next_power_of_2(N)"), ("grid", "M")),
        "scenarios.cases.reference", "reference(x) = x + 1",
        ("CUDA float32", "M、N 至少为 2", "N 不超过 1024", "行跨度至少为 N", "元素总数不超过 1000000", "显式尾部 mask"),
    ),
    "B": OperatorMeaning(
        "B", ("i",), "0 <= i < N",
        (InputMeaning("X", "X", ("N",), "x[i]"),),
        "Y", "out[i] = x[i] + 1",
        (("N", "x.size(0)"), ("B", "128"), ("grid", "ceil(N/B)")),
        "scenarios.cases.reference", "reference(x) = x + 1",
        ("CUDA float32", "N 至少为 1", "元素总数不超过 1000000", "显式尾部 mask"),
    ),
    "D": OperatorMeaning(
        "D", ("i",), "0 <= i < n_elements",
        (InputMeaning("X", "x_ptr", ("n_elements",), "x[i]"), InputMeaning("Y", "y_ptr", ("n_elements",), "y[i]")),
        "output_ptr", "out[i] = x[i] + y[i]",
        (("n_elements", "x.numel() = y.numel()"), ("BLOCK_SIZE", "128"), ("grid", "ceil(n_elements/BLOCK_SIZE)")),
        "scenarios.cases.reference", "reference(x, y) = x + y",
        ("两个输入均为同设备 CUDA float32", "形状一致", "长度至少为 1", "长度不超过 1000000", "显式尾部 mask"),
    ),
}
