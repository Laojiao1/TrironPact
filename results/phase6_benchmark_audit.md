# 第六阶段新增离线基准审计

生成时间：2026-09-25T18:09:57.453409+08:00

受限审计：通过。

| 检查 | 结果 |
| --- | --- |
| six_distinct_function_bodies | 通过 |
| all_ast_supported | 通过 |
| complete_machine_readable_specs | 通过 |
| call_bindings_audited | 通过 |
| unit_stride_candidates_proven | 通过 |
| workers_completed | 通过 |
| eligible_boundaries_correct | 通过 |
| ineligible_boundaries_witnessed | 通过 |
| shadow_candidates_match_labels | 通过 |

| Kernel | 布局 | 独立标签 | 运行观察 | 不一致元素 |
| --- | --- | --- | --- | --- |
| bench_relu | contiguous | eligible / candidate=True | correct | 0 |
| bench_relu | offset | eligible / candidate=True | correct | 0 |
| bench_relu | strided | ineligible / candidate=False | numeric_mismatch | 94 |
| bench_square | contiguous | eligible / candidate=True | correct | 0 |
| bench_square | offset | eligible / candidate=True | correct | 0 |
| bench_square | strided | ineligible / candidate=False | numeric_mismatch | 128 |
| bench_multiply | contiguous | eligible / candidate=True | correct | 0 |
| bench_multiply | offset | eligible / candidate=True | correct | 0 |
| bench_multiply | strided | ineligible / candidate=False | numeric_mismatch | 128 |
| bench_axpy | contiguous | eligible / candidate=True | correct | 0 |
| bench_axpy | offset | eligible / candidate=True | correct | 0 |
| bench_axpy | strided | ineligible / candidate=False | numeric_mismatch | 128 |
| bench_feature_scale | contiguous | eligible / candidate=True | correct | 0 |
| bench_feature_scale | offset | eligible / candidate=True | correct | 0 |
| bench_feature_scale | x_col_strided | ineligible / candidate=False | numeric_mismatch | 70 |
| bench_feature_scale | feature_strided | ineligible / candidate=False | numeric_mismatch | 69 |
| bench_feature_bias | contiguous | eligible / candidate=True | correct | 0 |
| bench_feature_bias | offset | eligible / candidate=True | correct | 0 |
| bench_feature_bias | x_col_strided | ineligible / candidate=False | numeric_mismatch | 70 |
| bench_feature_bias | feature_strided | ineligible / candidate=False | numeric_mismatch | 70 |

仅六个新增项目内离线 Kernel、固定一维与二维特征广播布局及本次 GPU 环境。标签来自独立逻辑索引、wrapper 调用绑定与实际 stride 审计；未编译 GuardPlan，也未扩大 Fast。
