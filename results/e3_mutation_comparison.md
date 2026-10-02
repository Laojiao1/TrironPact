# e3 变异方法公平对照

- 生成时间：2026-10-02T13:15:28.459681+08:00。
- 状态：`go`。
- 冻结协议：`{'methods': ('static_candidate', 'uniform_random', 'constrained_random', 'predicate_guided'), 'seeds': (17, 42, 73, 101, 211), 'attempt_budget': 14, 'worker_budget': 4, 'wall_clock_budget_ms': 120000.0, 'oracle': 'e3_real_risk_cases.json；每个唯一案例仅独立执行一次，比较阶段作不可变成本重放'}`。
- 贡献定位：谓词引导在多数基线的首次见证尝试数上更少。

| 方法 | 试验 | 首次见证成功 | 中位尝试数 | 中位 worker ms | 唯一见证 |
| --- | ---: | ---: | ---: | ---: | --- |
| static_candidate | 5 | 5 | 3 | 7210.250600999999 | ['liger_softmax_x_stride', 'triton_add_x_stride', 'unsloth_layernorm_affine_stride', 'unsloth_layernorm_x_stride'] |
| uniform_random | 5 | 5 | 1 | 6578.1958550000045 | ['liger_softmax_x_stride', 'triton_add_x_stride', 'unsloth_layernorm_affine_stride', 'unsloth_layernorm_x_stride'] |
| constrained_random | 5 | 5 | 4 | 6766.545522000001 | ['liger_softmax_x_stride', 'triton_add_x_stride', 'unsloth_layernorm_affine_stride', 'unsloth_layernorm_x_stride'] |
| predicate_guided | 5 | 5 | 2 | 6348.286799000008 | ['liger_softmax_x_stride', 'triton_add_x_stride', 'unsloth_layernorm_affine_stride', 'unsloth_layernorm_x_stride'] |

- 元数据快筛不计作经验风险见证；只有独立 GPU worker 的 `numeric_mismatch` 进入唯一见证数。
- 混合变异单列为 `mixed_unattributed`，不作单谓词因果归因；非法短 storage 只保留不可达理由，不送 GPU。
- `worker_elapsed_ms` 是不可变 Oracle 的成本重放，用于公平截断，不是 e4 端到端性能测量。
