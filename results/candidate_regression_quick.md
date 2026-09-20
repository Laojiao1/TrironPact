# TritonPact 受限契约与分派回归报告

> 生成时间：2026-09-20 15:13:11 CST
> 受限案例验收：**通过**
> A 模板内的动态候选修订：**已验证**

## 验收项

| 检查项 | 结果 |
| --- | --- |
| 四个变体进入 Access IR | 通过 |
| 语义规格与逐谓词来源可追溯 | 通过 |
| A 两组转置尺寸均出现数值错误 | 通过 |
| A Guard 拦截并正确回退 | 通过 |
| A2 非连续输入正确直通且连续化会复制 | 通过 |
| B 非整除边界正确直通 | 通过 |
| 未知类型安全回退 | 通过 |
| 官方教程双输入契约阻止错读 | 通过 |
| 关键输入重复稳定 | 通过 |
| 经验观察与静态放行分开记录 | 通过 |
| 单谓词补集触发候选修订并放行 | 通过 |

## 逐例结果

| 案例 | 尺寸 | 布局 | 检查方式 | 路径 | 结果 | Guard 依据 | 运行证据 | 说明 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A | 7×11 | contiguous | 初始 Guard | Fast | 正确 | Statically-Proven（放行） | Empirically-Validated |  |
| A | 7×11 | transpose | 原始 Fast | Raw Fast | 预期错读（74 处） | Unknown | Empirically-Validated | 首错 [0, 1]：预期 8.0，实际 2.0 |
| A | 5×13 | transpose | 原始 Fast | Raw Fast | 预期错读（60 处） | Unknown | Empirically-Validated | 首错 [0, 1]：预期 6.0，实际 2.0 |
| A | 7×11 | transpose | 初始 Guard | PyTorch Fallback | 正确 | Statically-Proven（拒绝） | Empirically-Validated |  |
| A | 7×11 | padded | 初始 Guard | PyTorch Fallback | 正确 | Statically-Proven（拒绝） | Empirically-Validated |  |
| A2 | 7×11 | padded | 初始 Guard | Fast | 正确 | Statically-Proven（放行） | Empirically-Validated | 输入非连续；contiguous() 会复制 |
| A2 | 5×13 | padded | 初始 Guard | Fast | 正确 | Statically-Proven（放行） | Empirically-Validated | 输入非连续；contiguous() 会复制 |
| A2 | 7×11 | transpose | 初始 Guard | PyTorch Fallback | 正确 | Unknown | Empirically-Validated |  |
| B | 127 | contiguous | 初始 Guard | Fast | 正确 | Statically-Proven（放行） | Empirically-Validated |  |
| B | 128 | contiguous | 初始 Guard | Fast | 正确 | Statically-Proven（放行） | Empirically-Validated |  |
| B | 129 | contiguous | 初始 Guard | Fast | 正确 | Statically-Proven（放行） | Empirically-Validated |  |
| B | 129 | offset | 初始 Guard | Fast | 正确 | Statically-Proven（放行） | Empirically-Validated |  |
| B | 129 | strided | 初始 Guard | PyTorch Fallback | 正确 | Statically-Proven（拒绝） | Empirically-Validated |  |
| A | 7×11 | unknown_dtype | 初始 Guard | PyTorch Fallback | 正确 | Unknown | Empirically-Validated |  |
| D | 127 | contiguous | 初始 Guard | Fast | 正确 | Statically-Proven（放行） | Empirically-Validated |  |
| D | 129 | contiguous | 初始 Guard | Fast | 正确 | Statically-Proven（放行） | Empirically-Validated |  |
| D | 129 | x_strided | 原始 Fast | Raw Fast | 预期错读（128 处） | Unknown | Empirically-Validated | 首错 [1]：预期 12.0，实际 11.0 |
| D | 129 | x_strided | 初始 Guard | PyTorch Fallback | 正确 | Statically-Proven（拒绝） | Empirically-Validated |  |
| D | 129 | y_strided | 原始 Fast | Raw Fast | 预期错读（128 处） | Unknown | Empirically-Validated | 首错 [1]：预期 3.0，实际 2.0 |
| D | 129 | y_strided | 初始 Guard | PyTorch Fallback | 正确 | Statically-Proven（拒绝） | Empirically-Validated |  |
| D | 129 | x_offset | 初始 Guard | Fast | 正确 | Statically-Proven（放行） | Empirically-Validated |  |
| D | 129 | unknown_dtype | 初始 Guard | PyTorch Fallback | 正确 | Unknown | Empirically-Validated |  |
| A | 1×11 | single_row | 原始 Fast | Raw Fast | 正确 | Unknown | Empirically-Validated |  |
| A | 7×1 | single_col | 原始 Fast | Raw Fast | 正确 | Unknown | Empirically-Validated |  |
| A | 1×11 | single_row | 初始 Guard | PyTorch Fallback | 正确 | Statically-Proven（拒绝） | Empirically-Validated |  |
| A | 7×1 | single_col | 初始 Guard | PyTorch Fallback | 正确 | Statically-Proven（拒绝） | Empirically-Validated |  |
| A | 1×11 | single_row | 修订后 Guard | Fast | 正确 | Statically-Proven（放行） | Empirically-Validated |  |
| A | 7×1 | single_col | 修订后 Guard | Fast | 正确 | Statically-Proven（放行） | Empirically-Validated |  |

`Statically-Proven` 是 Guard 在声明语义和支持域内的静态判断；仅当它放行时才有 Fast 资格。`Empirically-Validated` 只表示这一行的执行结果已观察到，不会单独授予 Fast 资格。


## 关键案例重复

表中次数包含逐例结果中的首次运行。

快速模式：关键案例各运行 1 次。

## 提取出的访问与布局条件

| 案例 | 输入 | 分析状态 | 输入地址偏移 | 读取 mask | 候选条件 |
| --- | --- | --- | --- | --- | --- |
| A | X | Supported | `tl.program_id(0) * B + tl.arange(0, B)` | `tl.program_id(0) * B + tl.arange(0, B) < M * N` | `size(0) == 1 或 X.stride(0) == size(1)；size(1) == 1 或 X.stride(1) == 1` |
| A2 | X | Supported | `tl.program_id(0) * S0 + tl.arange(0, B)` | `tl.arange(0, B) < N` | `X.stride(1) == 1` |
| B | X | Supported | `tl.program_id(0) * B + tl.arange(0, B)` | `tl.program_id(0) * B + tl.arange(0, B) < N` | `X.stride(0) == 1` |
| D | X | Supported | `tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` | `tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) < n_elements` | `X.stride(0) == 1` |
| D | Y | Supported | `tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)` | `tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) < n_elements` | `Y.stride(0) == 1` |

## 外部语义输入与谓词来源

算子语义、参数映射和参考实现是分析的外部输入；它们不由 Kernel AST 独自推得，也没有预填目标 stride 谓词。

| 案例 | 逻辑索引域 | 预期读取及形状 | 预期写入 | 启动参数映射 | 参考计算 | 参考入口 |
| --- | --- | --- | --- | --- | --- | --- |
| A | 0 <= row < M，0 <= col < N | X → x[row, col]，shape=('M', 'N') | out[row, col] = x[row, col] + 1 | M ← x.size(0)；N ← x.size(1)；B ← 128；grid ← ceil(M*N/B) | reference(x) = x + 1 | scenarios.cases.reference |
| A2 | 0 <= row < M，0 <= col < N | X → x[row, col]，shape=('M', 'N') | out[row, col] = x[row, col] + 1 | M ← x.size(0)；N ← x.size(1)；S0 ← x.stride(0)；B ← next_power_of_2(N)；grid ← M | reference(x) = x + 1 | scenarios.cases.reference |
| B | 0 <= i < N | X → x[i]，shape=('N',) | out[i] = x[i] + 1 | N ← x.size(0)；B ← 128；grid ← ceil(N/B) | reference(x) = x + 1 | scenarios.cases.reference |
| D | 0 <= i < n_elements | x_ptr → x[i]，shape=('n_elements',)；y_ptr → y[i]，shape=('n_elements',) | out[i] = x[i] + y[i] | n_elements ← x.numel() = y.numel()；BLOCK_SIZE ← 128；grid ← ceil(n_elements/BLOCK_SIZE) | reference(x, y) = x + y | scenarios.cases.reference |

每条布局条件对应的访问点和语义假设：

| 案例 | 布局条件 | 访存来源 | 语义读取 | 推导说明 |
| --- | --- | --- | --- | --- |
| A | size(0) == 1 或 X.stride(0) == size(1) | X 的 load：`scenarios/kernels.py:15` | x[row, col] | 线性地址与二维行优先逻辑索引对应 |
| A | size(1) == 1 或 X.stride(1) == 1 | X 的 load：`scenarios/kernels.py:15` | x[row, col] | 相邻列的读取地址差为 1 |
| A2 | X.stride(1) == 1 | X 的 load：`scenarios/kernels.py:25` | x[row, col] | 行步长来自输入，列地址以单位步长递增 |
| B | X.stride(0) == 1 | X 的 load：`scenarios/kernels.py:34` | x[i] | 线性地址与逐元素逻辑索引对应 |
| D | X.stride(0) == 1 | x_ptr 的 load：`scenarios/external_add.py:19` | x[i] | 线性地址与逐元素逻辑索引对应 |
| D | Y.stride(0) == 1 | y_ptr 的 load：`scenarios/external_add.py:20` | y[i] | 线性地址与逐元素逻辑索引对应 |

## 边界反例与候选修订

A 的初始候选在单行、单列输入上各误拒绝一个正确实例。隔离运行确认实例正确后，有限索引分析给出以下修订：

| 初始候选 | 修订后条件 |
| --- | --- |
| X.stride(0) == size(1) | size(0) == 1 或 X.stride(0) == size(1) |
| X.stride(1) == 1 | size(1) == 1 或 X.stride(1) == 1 |

边界证据：

| 输入尺寸 | 输入 stride | 只违反的原谓词 |
| --- | --- | --- |
| 1×11 | (14, 1) | stride(0) |
| 7×1 | (1, 3) | stride(1) |

推导依据：二维线性 idx=row*N+col；M=1 时 row 恒为 0，N=1 时 col 恒为 0，因此对应 stride 不影响任何读取地址。


## 结论边界

- `通过` 表示本报告中的受限语义、地址模板和输入域达到了 PoC 最低验收，不代表通用 Triton Kernel 安全。
- Case A 的原始 Fast 错读是预期的对照结果；Guard 应阻止相同错误布局进入 Fast。
- A 的单行、单列反例触发了真实候选修改；只有这两个实例的成功仍不足以证明整个补集，条件式还依赖报告列出的静态索引推导。
- Case B 的尾部 mask 在静态分析中已被识别，因此 `N % B == 0` 未进入契约；B 没有发生动态修订。
- Case D 取自 [Triton 官方向量加法教程](https://github.com/triton-lang/triton/blob/main/python/tutorials/01-vector-add.py)，验证了第二个输入指针与 load 的提取；它不是生产项目缺陷案例。
- 本报告记录数值正确性与路径；未测量端到端性能，也未对可选的对齐案例作结论。
- 机器数据把 `guard_basis`、`fast_eligible` 与 `observation_level` 分开保存；即使 Fallback 的输出正确，若静态条件不足，Fast 仍不放行。
