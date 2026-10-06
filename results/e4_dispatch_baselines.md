# e4a 系统分派基线

- 状态：`go`。
- 案例域：`{'valid_registered_kernels': 14, 'numeric_mismatch_violations': 2, 'safe_project_nonstandard': 1, 'safe_upstream_outside_frozen_domain': 1, 'total': 18}`。
- 边界：路径与复制账本来自 e4_workload_correctness 的实际执行；性能比较只允许使用全部正确且同语义的策略。

| 策略 | 正确 | 错误 | Fast/Raw | Fallback | 复制次数 | 复制字节 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| upstream | 17 | 1 | 15 | 0 | 2 | 65536 |
| handwritten_guard | 18 | 0 | 16 | 2 | 0 | 0 |
| always_copy | 18 | 0 | 18 | 0 | 18 | 378197 |
| always_fallback | 18 | 0 | 0 | 18 | 0 | 0 |
| static_no_refinement | 16 | 2 | 18 | 0 | 0 | 0 |
| tritonpact | 18 | 0 | 15 | 3 | 0 | 0 |
| deterministic_no_cost | 18 | 0 | 15 | 3 | 0 | 0 |

## 选择规则

- **upstream**：固定上游 wrapper；Liger 防御复制，Triton 教程调用无同等布局防御。
- **handwritten_guard**：人工检查所有输入最后一维步长为 1；违约时使用同一 PyTorch reference。
- **always_copy**：每个案例显式 clone/物化后执行；所有输入字节均计入。
- **always_fallback**：全部执行登记的独立 PyTorch reference。
- **static_no_refinement**：只核对 shape/dtype，不核对物理 stride；两个实际 raw mismatch 均保留。
- **tritonpact**：冻结 Guard；False/Unknown 使用 reference，未在线标定成本表。
- **deterministic_no_cost**：关闭成本表后的确定性安全策略；本阶段与 TritonPact 选择一致。
