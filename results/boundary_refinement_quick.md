# TritonPact 第四阶段边界证据与精化报告

> 生成时间：2026-09-20T20:40:13.240958+08:00  
> 验收：通过；快速运行  
> 新证据与精化建议只作离线/影子分析，旧 Guard 未接入。

## 检查

| 检查项 | 结果 |
| --- | --- |
| 变异覆盖八个案例且每例独立隔离 | 通过 |
| 执行分类完整且无未解释故障 | 通过 |
| 已知错读与正确边界分开 | 通过 |
| 指针提示不满足时不强制运行 | 通过 |
| 末端与越界一格只作元数据影子 | 通过 |
| SMT 冗余 fixture 不升级真实义务 | 通过 |
| 历史过强见证具独立静态修订理由 | 通过 |
| 静态候选没有未处理的满足集错读 | 通过 |
| 第三阶段基线仍通过 | 通过 |

## 逐例变异

| ID | 目标 | 候选变化 | 混合 | Oracle | 旧 Guard 对照 | 建议 |
| --- | --- | --- | --- | --- | --- | --- |
| `A:transpose:aca6a86f0e0a` | shape_stride | [0, 1] | 是 | numeric_mismatch | PyTorch Fallback | review_understrong_or_expected_rejection |
| `A2:padded:855694dec116` | shape_stride | [] | 否 | correct | Fast | keep_observation_only |
| `holdout_matrix:transpose:8fce31b5aa78` | shape_stride | [] | 否 | correct | Unsupported | keep_observation_only |
| `B:tile_plus:526b0ef95248` | tile_mask | [] | 否 | correct | Fast | keep_observation_only |
| `holdout_vector:stride_two:386761ca5839` | shape_stride | [] | 否 | correct | Unsupported | keep_observation_only |
| `pointer:tile_exact:f5bf555d065f` | tile_mask | [] | 否 | correct | NotObserved | keep_observation_only |
| `index:offset:6f4077f23470` | alignment | [] | 否 | correct | NotObserved | keep_observation_only |
| `D:x_stride:b26f771ef23e` | shape_stride | [0] | 否 | numeric_mismatch | PyTorch Fallback | review_understrong_or_expected_rejection |

## 计数与逻辑边界

- 分类计数：`{"correct": 6, "numeric_mismatch": 2}`。
- Span 末端：True；越界一格：False；仅元数据影子，无非法 GPU 视图运行。
- 历史 A singleton 过强见证：True；静态理由：二维线性 idx=row*N+col；M=1 时 row 恒为 0，N=1 时 col 恒为 0，因此对应 stride 不影响任何读取地址。。
- SMT 合成冗余 fixture 删除索引：[1]；真实候选删除：[]。
- 正确补集与错读只构成所测输入的证据；混合变异不作单谓词因果归因。Z3 的合成 fixture 不提升真实动态指针对齐义务。
- 完整输入配方、实际地址余数、参考比较、子进程诊断、逐候选影子值及 SMT 查询见同名 JSON。
