# TritonPact 第四阶段边界证据与精化报告

> 生成时间：2026-09-20T20:36:38.546300+08:00  
> 验收：通过；完整运行  
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
| `A:contiguous:e71335223979` | shape_stride | [] | 否 | correct | Fast | keep_observation_only |
| `A:padded:6768a1a63380` | shape_stride | [0] | 否 | numeric_mismatch | PyTorch Fallback | review_understrong_or_expected_rejection |
| `A:transpose:aca6a86f0e0a` | shape_stride | [0, 1] | 是 | numeric_mismatch | PyTorch Fallback | review_understrong_or_expected_rejection |
| `A:single_row:fca10b892a37` | shape_stride | [] | 否 | correct | PyTorch Fallback | keep_observation_only |
| `A:single_col:75fdc5142d9b` | shape_stride | [] | 否 | correct | PyTorch Fallback | keep_observation_only |
| `A2:contiguous:a668b3ad3da3` | shape_stride | [] | 否 | correct | NotObserved | keep_observation_only |
| `A2:padded:855694dec116` | shape_stride | [] | 否 | correct | Fast | keep_observation_only |
| `A2:transpose:75182b01202a` | shape_stride | [0] | 否 | numeric_mismatch | PyTorch Fallback | review_understrong_or_expected_rejection |
| `A2:single_row:913abeedd9ba` | shape_stride | [] | 否 | correct | NotObserved | keep_observation_only |
| `A2:single_col:52b390ae9544` | shape_stride | [] | 否 | correct | NotObserved | keep_observation_only |
| `holdout_matrix:contiguous:718505c37553` | shape_stride | [] | 否 | correct | Unsupported | keep_observation_only |
| `holdout_matrix:padded:f796dd99eafb` | shape_stride | [] | 否 | correct | Unsupported | keep_observation_only |
| `holdout_matrix:transpose:8fce31b5aa78` | shape_stride | [] | 否 | correct | Unsupported | keep_observation_only |
| `holdout_matrix:single_row:f511d6410949` | shape_stride | [] | 否 | correct | Unsupported | keep_observation_only |
| `holdout_matrix:single_col:8d49ee5f3a3f` | shape_stride | [] | 否 | correct | Unsupported | keep_observation_only |
| `B:tile_minus:b58d9a1b3d2e` | tile_mask | [] | 否 | correct | Fast | keep_observation_only |
| `B:tile_exact:4f03f7e2a1e9` | tile_mask | [] | 否 | correct | Fast | keep_observation_only |
| `B:tile_plus:526b0ef95248` | tile_mask | [] | 否 | correct | Fast | keep_observation_only |
| `B:offset:235661881bf9` | span | [] | 否 | correct | Fast | keep_observation_only |
| `B:stride_two:286b0b4b1dea` | shape_stride | [0] | 否 | numeric_mismatch | PyTorch Fallback | review_understrong_or_expected_rejection |
| `holdout_vector:tile_minus:66fa342f8bdf` | tile_mask | [] | 否 | correct | Unsupported | keep_observation_only |
| `holdout_vector:tile_exact:60857c00122f` | tile_mask | [] | 否 | correct | Unsupported | keep_observation_only |
| `holdout_vector:tile_plus:90e42579079d` | tile_mask | [] | 否 | correct | Unsupported | keep_observation_only |
| `holdout_vector:offset:04fca8cab476` | span | [] | 否 | correct | Unsupported | keep_observation_only |
| `holdout_vector:stride_two:386761ca5839` | shape_stride | [] | 否 | correct | Unsupported | keep_observation_only |
| `pointer:tile_minus:c447382a17a7` | tile_mask | [] | 否 | correct | NotObserved | keep_observation_only |
| `pointer:tile_exact:f5bf555d065f` | tile_mask | [] | 否 | correct | NotObserved | keep_observation_only |
| `pointer:tile_plus:c755dd88c039` | tile_mask | [] | 否 | correct | NotObserved | keep_observation_only |
| `pointer:offset:90b9da6f7d36` | alignment | [0] | 否 | preflight_skip | NotObserved | inconclusive |
| `index:tile_minus:3f9cf1065781` | tile_mask | [] | 否 | correct | NotObserved | keep_observation_only |
| `index:tile_exact:23be33f3514e` | tile_mask | [] | 否 | correct | NotObserved | keep_observation_only |
| `index:tile_plus:4bfffb935d7c` | tile_mask | [] | 否 | correct | NotObserved | keep_observation_only |
| `index:offset:6f4077f23470` | alignment | [] | 否 | correct | NotObserved | keep_observation_only |
| `D:contiguous:86b24b16b992` | shape_stride | [] | 否 | correct | Fast | keep_observation_only |
| `D:x_stride:b26f771ef23e` | shape_stride | [0] | 否 | numeric_mismatch | PyTorch Fallback | review_understrong_or_expected_rejection |
| `D:y_stride:c820dcbcfdde` | shape_stride | [1] | 否 | numeric_mismatch | Fast | review_understrong_or_expected_rejection |

## 计数与逻辑边界

- 分类计数：`{"correct": 29, "numeric_mismatch": 6, "preflight_skip": 1}`。
- Span 末端：True；越界一格：False；仅元数据影子，无非法 GPU 视图运行。
- 历史 A singleton 过强见证：True；静态理由：二维线性 idx=row*N+col；M=1 时 row 恒为 0，N=1 时 col 恒为 0，因此对应 stride 不影响任何读取地址。。
- SMT 合成冗余 fixture 删除索引：[1]；真实候选删除：[]。
- 正确补集与错读只构成所测输入的证据；混合变异不作单谓词因果归因。Z3 的合成 fixture 不提升真实动态指针对齐义务。
- 完整输入配方、实际地址余数、参考比较、子进程诊断、逐候选影子值及 SMT 查询见同名 JSON。
