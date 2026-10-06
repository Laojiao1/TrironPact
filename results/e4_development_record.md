# e4a 开发记录

> e4a 只覆盖真实工作负载集成、系统基线和 WSL2 诊断。两套原生 Linux GPU 的 e4b 尚未执行，不能据此授予 `go_e4=true`。

## 工作块 1：工作负载与函数体清单冻结

- 命令：`python -m pytest test/test_e4_manifest.py -q`
- 结果：退出码 0，`2 passed`。
- 命令：`python -m bench.e4.manifest`
- 结果：退出码 0，`go_manifest=true`；2 个工作负载、14 个不同函数体。
- 证据：`results/e4_workload_manifest.md/.json`。
- 边界：4 个 holdout Kernel 只作冻结下游评估，不反馈修改 e2 规则。

## 工作块 2：工作负载正确性与布局边界

- 命令：`python -m pytest test/test_e4_manifest.py test/test_e4_workloads.py test/test_e4_correctness.py -q`
- 结果：退出码 0，`6 passed`。
- 命令：`python -m bench.e4.correctness`
- 结果：退出码 0，`go_correctness=true`；两个工作负载、14 个不同函数体实际执行，14 条冻结 Fast 路径正确，2 个数值错读布局均被阻止，已知误放行 0。
- 非标准布局：历史项目 A2 padded-row 安全直通且不复制；Liger softmax row-padding 数值正确但超出 e2 exact-stride 声明域，保守拒绝并单列为能力缺口。
- 证据：`results/e4_workload_correctness.md/.json`。

## 工作块 3：系统基线与 WSL2 诊断

- 命令：`python -m pytest test/test_e4_baselines.py -q`
- 结果：退出码 0，`1 passed`。
- 命令：`python -m bench.e4.baselines`
- 结果：退出码 0，`go_baselines=true`；7 类策略使用同一 18 案例分母，复制、Fallback 和错误选择均计入。
- 命令：`python -m pytest test/test_e4_diagnostics.py -q`
- 结果：退出码 0，`2 passed`。
- 命令：`python -m bench.e4.diagnostics`
- 结果：退出码 0，`go_diagnostics=true`；20 轮固定种子交错样本，首次编译与稳态分离，Guard/复制/Kernel/Fallback 分项保存。
- 观察：该极小 Inductor 图上完整 TritonPact 的配对中位延迟比约 2.29，95% bootstrap 区间约 `[2.08, 2.55]`，即当前单机诊断显示明确开销而非加速。
- 边界：WSL2 RTX 5060 Laptop 单机结果不形成原生 Linux、跨硬件或稳定性能收益结论。
- 证据：`results/e4_dispatch_baselines.md/.json`、`results/e4_wsl_diagnostics.md/.json`。

## 工作块 4：e4a 总验收

- 命令：`python -m pytest test/test_e4_*.py -q`
- 结果：退出码 0，`10 passed`。
- 命令：`python -m bench.e4.validate`
- 结果：退出码 0；完整回归 `144 passed`，旧隔离检查 11/11，通过全部 e1/e2/e3 指纹和工作区范围核对。
- 状态：`go_e4_integration=true`，`go_e4_cross_hardware=false`，`go_e4=false`。
- 源码指纹：`1553de651bce580f2821e641fde905b9c2f58a9f26c46c123b4b7b994289e5e0`。
- 证据：`results/e4_regression.md/.json`、`results/e4_regression_isolation.md/.json`、`results/e4_cross_hardware_performance.md/.json`。

e4a 已完成。e4b 因两套原生 Linux GPU 尚未取得而保持 `pending_resource`，没有运行正式跨硬件实验，也没有授予 e4 总体 Go。

