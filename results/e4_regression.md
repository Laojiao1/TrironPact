# e4 阶段状态与验收记录

生成时间：2026-10-12T09:47:05.264066+08:00
状态：`go`；`go_e4_integration=True`，`go_e4_cross_hardware=True`，`go_e4=True`。
源码指纹：`a74ab630744c42a1208d3df444fdc9d8f2f9f7381b20c026ae79e9621620da47`。

计数：`{'workloads': 2, 'integration_steps': 14, 'distinct_kernels': 14, 'by_source': {'pytorch': 4, 'triton': 1, 'liger': 7, 'unsloth': 2}, 'holdout_kernels': 4, 'workloads_complete': 2, 'distinct_kernels_executed': 14, 'fast_paths': 14, 'known_false_allows': 0, 'blocked_numeric_mismatches': 2, 'safe_nonstandard_fast_paths': 1, 'safe_outside_frozen_domain_rejections': 1, 'baseline_policies': 7, 'native_linux_environments': 2}`。

## 核验项

| 核验 | 结果 |
| --- | --- |
| e1_fingerprint_current | 通过 |
| e2_go_and_rule_fingerprint_current | 通过 |
| e3_go_and_source_fingerprint_current | 通过 |
| two_traceable_workloads_and_ten_kernels | 通过 |
| workload_correctness_and_zero_known_false_allows | 通过 |
| safe_nonstandard_and_blocked_violation | 通过 |
| fair_system_baselines | 通过 |
| wsl_diagnostics_complete_and_scoped | 通过 |
| full_regression_and_legacy_isolation | 通过 |
| no_e2_e3_or_online_fast_changes | 通过 |
| e4b_status_consistent | 通过 |

## 结论边界

- **integration**：两个真实轨迹和 14 个不同冻结函数体完成可追溯接入。
- **safety**：固定案例中已知误放行 0；2 个数值错读布局被阻止，Fallback 正确。
- **nonstandard**：项目 A2 padded-row 安全直通；Liger row-padding 因 e2 exact-stride 域被保守拒绝并单列缺口。
- **performance**：RTX 4090 与 RTX 5080 原生 Linux 正式报告均完整；所测极小图的完整分派区间稳定显示额外开销，不宣称加速或外推到大张量与完整模型。
- **overall**：e4a 与两套原生 Linux e4b 均通过，e4 总体 Go；性能结论保持逐环境、逐工作负载边界。

## 回归

- `python test/run_tests.py --quick --output /mnt/f/Project/Paper/Code/TritonPact/results/e4_regression_isolation.md`：退出码 0；163 passed in 95.79s (0:01:35)；旧隔离 11/11。
