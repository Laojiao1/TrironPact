# TritonPact 第三阶段候选契约提取报告

> 生成时间：2026-09-20T15:25:40.185049+08:00  
> 验收：**通过**  
> 候选仅作离线与影子核对；旧 Guard 未接入。

## 计数与检查

- 已生成记录（含拒绝义务）：43；静态充分规则：23；Unproven：1；Unknown：19。

| 检查 | 结果 |
| --- | --- |
| 六个案例由同一 IR 与候选规则处理 | 通过 |
| 三类候选有逐访问来源和绑定 | 通过 |
| 影子样例的 shape 与 span 条件可求值 | 通过 |
| 提示区分指针与索引且未知向量不推指针对齐 | 通过 |
| 错误绑定与缺失信息拒绝 | 通过 |
| 旧隔离回归通过且留出名称无 Fast 路径 | 通过 |

## 逐条候选

| 案例 | 类型 | 访存来源 | 条件或义务 | 用途 | 状态 | 影子值 | 推导理由 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | shape_stride | `scenarios/kernels.py:15` load `X` | `(X.size(0) == 1 或 X.stride(0) == X.size(1))` | Semantics | Statically-Proven | True | 扁平地址按列数 N 换行；单行时行步长无作用 |
| A | shape_stride | `scenarios/kernels.py:15` load `X` | `(X.size(1) == 1 或 X.stride(1) == 1)` | Semantics | Statically-Proven | True | 同一行相邻列地址差为 1；单列时列步长无作用 |
| A | span | `scenarios/kernels.py:15` load `X` | `X[0..(((M*N)-1)+0)] ∈ storage (4 B/elem)` | Safety | Statically-Proven | True | mask 限定 0<=idx<M*N；该访存偏移为 idx+0，实际启用范围为 [0, total-1+0] 元素；storage_offset 按元素计，端点乘元素宽度后与 storage_nbytes 比较；data_ptr 不重复叠加 offset |
| A | span | `scenarios/kernels.py:16` store `Y` | `OUT[0..(((M*N)-1)+0)] ∈ storage (4 B/elem)` | Safety | Statically-Proven | True | mask 限定 0<=idx<M*N；该访存偏移为 idx+0，实际启用范围为 [0, total-1+0] 元素；storage_offset 按元素计，端点乘元素宽度后与 storage_nbytes 比较；data_ptr 不重复叠加 offset |
| A2 | shape_stride | `scenarios/kernels.py:25` load `X` | `(X.size(1) == 1 或 X.stride(1) == 1)` | Semantics | Statically-Proven | True | 行系数由已核对的 stride(0) 提供；列地址差为 1，单列时列步长无作用 |
| A2 | span | `scenarios/kernels.py:25` load `X` | `X[0..((((M-1)*S0)+(N-1))+0)] ∈ storage (4 B/elem)` | Safety | Statically-Proven | True | grid 覆盖 0<=pid<M，mask 限定 0<=lane<N；行列系数需非负，实际启用范围为 [0, (M-1)*S0+N-1+0] 元素；storage_offset 按元素计，端点乘元素宽度后与 storage_nbytes 比较；data_ptr 不重复叠加 offset |
| A2 | span | `scenarios/kernels.py:26` store `Y` | `OUT[0..((((M-1)*N)+(N-1))+0)] ∈ storage (4 B/elem)` | Safety | Statically-Proven | True | grid 覆盖 0<=pid<M，mask 限定 0<=lane<N；行列系数需非负，实际启用范围为 [0, (M-1)*N+N-1+0] 元素；storage_offset 按元素计，端点乘元素宽度后与 storage_nbytes 比较；data_ptr 不重复叠加 offset |
| B | shape_stride | `scenarios/kernels.py:34` load `X` | `(X.size(0) == 1 或 X.stride(0) == 1)` | Semantics | Statically-Proven | True | mask 限定 0<=i<N；相邻逻辑元素地址差为 1，单元素时步长无作用 |
| B | span | `scenarios/kernels.py:34` load `X` | `X[0..((N-1)+0)] ∈ storage (4 B/elem)` | Safety | Statically-Proven | True | mask 限定 0<=idx<N；该访存偏移为 idx+0，实际启用范围为 [0, total-1+0] 元素；storage_offset 按元素计，端点乘元素宽度后与 storage_nbytes 比较；data_ptr 不重复叠加 offset |
| B | span | `scenarios/kernels.py:35` store `Y` | `OUT[0..((N-1)+0)] ∈ storage (4 B/elem)` | Safety | Statically-Proven | True | mask 限定 0<=idx<N；该访存偏移为 idx+0，实际启用范围为 [0, total-1+0] 元素；storage_offset 按元素计，端点乘元素宽度后与 storage_nbytes 比较；data_ptr 不重复叠加 offset |
| D | shape_stride | `scenarios/external_add.py:19` load `x_ptr` | `(X.size(0) == 1 或 X.stride(0) == 1)` | Semantics | Statically-Proven | True | mask 限定 0<=i<N；相邻逻辑元素地址差为 1，单元素时步长无作用 |
| D | shape_stride | `scenarios/external_add.py:20` load `y_ptr` | `(Y.size(0) == 1 或 Y.stride(0) == 1)` | Semantics | Statically-Proven | True | mask 限定 0<=i<N；相邻逻辑元素地址差为 1，单元素时步长无作用 |
| D | span | `scenarios/external_add.py:19` load `x_ptr` | `X[0..((n_elements-1)+0)] ∈ storage (4 B/elem)` | Safety | Statically-Proven | True | mask 限定 0<=idx<n_elements；该访存偏移为 idx+0，实际启用范围为 [0, total-1+0] 元素；storage_offset 按元素计，端点乘元素宽度后与 storage_nbytes 比较；data_ptr 不重复叠加 offset |
| D | span | `scenarios/external_add.py:20` load `y_ptr` | `Y[0..((n_elements-1)+0)] ∈ storage (4 B/elem)` | Safety | Statically-Proven | True | mask 限定 0<=idx<n_elements；该访存偏移为 idx+0，实际启用范围为 [0, total-1+0] 元素；storage_offset 按元素计，端点乘元素宽度后与 storage_nbytes 比较；data_ptr 不重复叠加 offset |
| D | span | `scenarios/external_add.py:22` store `output_ptr` | `OUT[0..((n_elements-1)+0)] ∈ storage (4 B/elem)` | Safety | Statically-Proven | True | mask 限定 0<=idx<n_elements；该访存偏移为 idx+0，实际启用范围为 [0, total-1+0] 元素；storage_offset 按元素计，端点乘元素宽度后与 storage_nbytes 比较；data_ptr 不重复叠加 offset |
| holdout_vector | shape_stride | `scenarios/holdouts.py:14` load `X` | `X.stride(0) == S` | Semantics | Statically-Proven | True | 一维读取地址为 i*已核对的输入 stride(0)，按逻辑索引逐元素读取 |
| holdout_vector | span | `scenarios/holdouts.py:14` load `X` | `X[0..((N-1)*S)] ∈ storage (4 B/elem)` | Safety | Statically-Proven | True | mask 限定 0<=idx<N；读取偏移为 idx*S，非负步长时实际启用范围为 [0, (N-1)*S] 元素；storage_offset 按元素计，端点乘元素宽度后与 storage_nbytes 比较；data_ptr 不重复叠加 offset |
| holdout_vector | span | `scenarios/holdouts.py:15` store `Y` | `OUT[0..((N-1)+0)] ∈ storage (4 B/elem)` | Safety | Statically-Proven | True | mask 限定 0<=idx<N；该访存偏移为 idx+0，实际启用范围为 [0, total-1+0] 元素；storage_offset 按元素计，端点乘元素宽度后与 storage_nbytes 比较；data_ptr 不重复叠加 offset |
| holdout_matrix | shape_stride | `scenarios/holdouts.py:23` load `X` | `X.stride(0) == S0` | Semantics | Statically-Proven | True | 逐行读取地址的行系数来自已核对的输入 stride(0) |
| holdout_matrix | shape_stride | `scenarios/holdouts.py:23` load `X` | `X.stride(1) == S1` | Semantics | Statically-Proven | True | 逐列读取地址的列系数来自已核对的输入 stride(1) |
| holdout_matrix | span | `scenarios/holdouts.py:23` load `X` | `X[0..((((M-1)*S0)+((N-1)*S1))+0)] ∈ storage (4 B/elem)` | Safety | Statically-Proven | True | grid 覆盖 0<=pid<M，mask 限定 0<=lane<N；行列系数需非负，实际启用范围为 [0, (M-1)*S0+(N-1)*S1+0] 元素；storage_offset 按元素计，端点乘元素宽度后与 storage_nbytes 比较；data_ptr 不重复叠加 offset |
| holdout_matrix | span | `scenarios/holdouts.py:24` store `Y` | `OUT[0..((((M-1)*N)+(N-1))+0)] ∈ storage (4 B/elem)` | Safety | Statically-Proven | True | grid 覆盖 0<=pid<M，mask 限定 0<=lane<N；行列系数需非负，实际启用范围为 [0, (M-1)*N+N-1+0] 元素；storage_offset 按元素计，端点乘元素宽度后与 storage_nbytes 比较；data_ptr 不重复叠加 offset |
| pointer | alignment | `scenarios/alignment_fixtures.py:13` load `X` | `X.effective_ptr % 16 == 0` | Optimization | Unproven | True | scenarios/alignment_fixtures.py:10 的 tl.multiple_of(X, 16) 作用于有效基址；指针模数按字节计，data_ptr 已含视图偏移，不再叠加 storage_offset；动态地址尚未静态保证 |
| index | alignment | `scenarios/alignment_fixtures.py:22` load `X` | `B % 16 == 0` | Optimization | Statically-Proven | True | scenarios/alignment_fixtures.py:19 的 tl.multiple_of(tl.program_id(0) * B, 16) 仅约束索引组首 pid(0)*B 的元素偏移；不推出任何 Tensor 的有效指针对齐 |
| reject:wrong_grid | shape_stride | `scenarios/kernels.py:34` load `X` | `义务：add(lane(param(3)), mul(param(3), pid(0)))` | Semantics | Unknown | — | grid 未匹配受支持的调用绑定 |
| reject:wrong_grid | span | `scenarios/kernels.py:34` load `X` | `义务：add(lane(param(3)), mul(param(3), pid(0)))` | Safety | Unknown | — | grid 未匹配受支持的调用绑定 |
| reject:wrong_grid | span | `scenarios/kernels.py:35` store `Y` | `义务：add(lane(param(3)), mul(param(3), pid(0)))` | Safety | Unknown | — | grid 未匹配受支持的调用绑定 |
| reject:wrong_pointer | shape_stride | `scenarios/kernels.py:34` load `X` | `义务：add(lane(param(3)), mul(param(3), pid(0)))` | Semantics | Unknown | — | 指针身份与独立语义或 Kernel 签名冲突 |
| reject:wrong_pointer | span | `scenarios/kernels.py:34` load `X` | `义务：add(lane(param(3)), mul(param(3), pid(0)))` | Safety | Unknown | — | 指针身份与独立语义或 Kernel 签名冲突 |
| reject:wrong_pointer | span | `scenarios/kernels.py:35` store `Y` | `义务：add(lane(param(3)), mul(param(3), pid(0)))` | Safety | Unknown | — | 指针身份与独立语义或 Kernel 签名冲突 |
| reject:missing_width | shape_stride | `scenarios/kernels.py:34` load `X` | `义务：add(lane(param(3)), mul(param(3), pid(0)))` | Semantics | Unknown | — | 输出元素宽度缺失或与访存点不一致 |
| reject:missing_width | span | `scenarios/kernels.py:34` load `X` | `义务：add(lane(param(3)), mul(param(3), pid(0)))` | Safety | Unknown | — | 输出元素宽度缺失或与访存点不一致 |
| reject:missing_width | span | `scenarios/kernels.py:35` store `Y` | `义务：add(lane(param(3)), mul(param(3), pid(0)))` | Safety | Unknown | — | 输出元素宽度缺失或与访存点不一致 |
| reject:wrong_output | shape_stride | `scenarios/kernels.py:34` load `X` | `义务：add(lane(param(3)), mul(param(3), pid(0)))` | Semantics | Unknown | — | 新建输出形状与独立逻辑形状不符 |
| reject:wrong_output | span | `scenarios/kernels.py:34` load `X` | `义务：add(lane(param(3)), mul(param(3), pid(0)))` | Safety | Unknown | — | 新建输出形状与独立逻辑形状不符 |
| reject:wrong_output | span | `scenarios/kernels.py:35` store `Y` | `义务：add(lane(param(3)), mul(param(3), pid(0)))` | Safety | Unknown | — | 新建输出形状与独立逻辑形状不符 |
| reject:wrong_mask | shape_stride | `<provided>:5` load `X` | `义务：add(lane(param(3)), mul(param(3), pid(0)))` | Semantics | Unknown | — | 各访存点的 mask 不一致 |
| reject:wrong_mask | span | `<provided>:5` load `X` | `义务：add(lane(param(3)), mul(param(3), pid(0)))` | Safety | Unknown | — | 各访存点的 mask 不一致 |
| reject:wrong_mask | span | `<provided>:6` store `Y` | `义务：add(lane(param(3)), mul(param(3), pid(0)))` | Safety | Unknown | — | 各访存点的 mask 不一致 |
| reject:wrong_stride | shape_stride | `scenarios/kernels.py:25` load `X` | `义务：add(lane(param(5)), mul(param(4), pid(0)))` | Semantics | Unknown | — | shape/stride/Tile 标量与独立语义或 Kernel 签名冲突 |
| reject:wrong_stride | span | `scenarios/kernels.py:25` load `X` | `义务：add(lane(param(5)), mul(param(4), pid(0)))` | Safety | Unknown | — | shape/stride/Tile 标量与独立语义或 Kernel 签名冲突 |
| reject:wrong_stride | span | `scenarios/kernels.py:26` store `Y` | `义务：add(lane(param(5)), mul(param(3), pid(0)))` | Safety | Unknown | — | shape/stride/Tile 标量与独立语义或 Kernel 签名冲突 |
| reject:vector_hint | alignment | `<provided>:6` load `X` | `义务：add(lane(param(3)), mul(param(3), pid(0)))` | Optimization | Unknown | — | <provided>:4 的 tl.multiple_of(idx, 16) 的组首不是受支持的标量 pid(0)*Tile 形式 |

完整 JSON 保留每个访问点的原始/规范化地址、mask、逻辑索引、调用绑定、支持域和未降级义务。影子值来自报告中列明的代表性元数据，不能代表所有运行输入。

## 旧 Guard 路径对照

| 样例 | 路径 |
| --- | --- |
| A2_padded | Fast |
| B_offset | Fast |
| D_x_offset | Fast |
| holdout_vector | Unsupported |
| holdout_matrix | Unsupported |

## 拒绝边界

| 变体 | shape/stride | span | 原因 |
| --- | --- | --- | --- |
| missing_semantics | Unknown | Unknown | 缺少独立算子语义; 缺少独立算子语义 |
| wrong_grid | Unknown | Unknown | grid 未匹配受支持的调用绑定; grid 未匹配受支持的调用绑定 |
| wrong_pointer | Unknown | Unknown | 指针身份与独立语义或 Kernel 签名冲突; 指针身份与独立语义或 Kernel 签名冲突 |
| missing_width | Unknown | Unknown | 输出元素宽度缺失或与访存点不一致; 输出元素宽度缺失或与访存点不一致 |
| wrong_output | Unknown | Unknown | 新建输出形状与独立逻辑形状不符; 新建输出形状与独立逻辑形状不符 |
| wrong_mask | Unknown | Unknown | 各访存点的 mask 不一致; 各访存点的 mask 不一致 |
| wrong_stride | Unknown | Unknown | shape/stride/Tile 标量与独立语义或 Kernel 签名冲突; shape/stride/Tile 标量与独立语义或 Kernel 签名冲突 |
| vector_hint | — | — | alignment Unknown：提示作用对象已分类；不授予 Fast 资格；未生成指针对齐条件 |

## 证据边界

- `Statically-Proven` 表示在声明的独立语义、调用绑定和受支持 IR 规则下，该候选是充分的条件形式；影子值为当前样例的元数据求值。
- 隔离数值和旧分派路径以单独的第三阶段 `candidate_regression.md/.json` 为准；有限运行不构成一般性安全证明。
- 间接索引、动态循环、未知 mask/grid、负步长、未知元素宽度和复杂提示形式仍为 Unknown/Unsupported。
- 新候选没有接入 Fast、Guard 或新的分派路径。
