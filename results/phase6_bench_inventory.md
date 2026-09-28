# 第六阶段基准选型摸底

生成时间：2026-09-25T18:07:23.196144+08:00

登记 15 个不同函数体；AST 可解析 14，Unknown 0，Unsupported 1；已固定且审计许可的外部来源 2 个。
五层分母：登记 15，AST 可解析 14，候选完整 14，Guard 可用 8，Fast 实际可行 8。
正向可解析语义类计数：{'elementwise_mapping': 9, 'feature_broadcast': 2, 'layout_2d': 3}。

| ID | 角色 | 语义类 | 来源 | AST | 候选 | Guard | Fast |
| --- | --- | --- | --- | --- | --- | --- | --- |
| A | positive | layout_2d | project | Supported | complete | available | feasible |
| A2 | positive | layout_2d | project | Supported | complete | available | feasible |
| B | positive | elementwise_mapping | project | Supported | complete | available | feasible |
| D | positive | elementwise_mapping | triton-lang/triton | Supported | complete | available | feasible |
| holdout_vector | positive | elementwise_mapping | project | Supported | complete | available | feasible |
| holdout_matrix | positive | layout_2d | project | Supported | complete | available | feasible |
| pointer_hint | positive | elementwise_mapping | project | Supported | complete | available | feasible |
| index_hint | positive | elementwise_mapping | project | Supported | complete | available | feasible |
| bench_relu | positive | elementwise_mapping | project | Supported | shadow_complete | not_registered | not_eligible |
| bench_square | positive | elementwise_mapping | project | Supported | shadow_complete | not_registered | not_eligible |
| bench_multiply | positive | elementwise_mapping | project | Supported | shadow_complete | not_registered | not_eligible |
| bench_axpy | positive | elementwise_mapping | project | Supported | shadow_complete | not_registered | not_eligible |
| bench_feature_scale | positive | feature_broadcast | project | Supported | shadow_complete | not_registered | not_eligible |
| bench_feature_bias | positive | feature_broadcast | project | Supported | shadow_complete | not_registered | not_eligible |
| vllm_expand_idx | challenge | index_expansion | vllm-project/vllm | Unsupported | unsupported | unsupported | unsupported |

仅核对清单、源码指纹和 AST 可解析性；positive/challenge 分列。未执行语义/绑定/Guard/GPU 核验，外部未固定版本不计入合格来源。

## 待核查来源（不计入基准）

- pytorch/pytorch：https://github.com/pytorch/pytorch/issues/134372；含生成的点式 Triton 样例；待固定原始提交、审计语义和解析，不计入基准
- linkedin/Liger-Kernel：https://github.com/linkedin/Liger-Kernel/blob/40a9d8a61bdc546f4e70220619d4db9760fa80b1/src/liger_kernel/ops/swiglu.py；固定源码的 SwiGLU forward 使用指针就地加法与 cast；现有单轴仿射解析器不支持，不计入基准
- triton-lang/triton：https://github.com/triton-lang/triton/blob/main/python/tutorials/02-fused-softmax.py；规约与循环挑战样例；当前解析器不支持，不计入基准
