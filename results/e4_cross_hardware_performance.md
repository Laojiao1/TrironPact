# e4b 原生 Linux 跨硬件复核

- 状态：`go`；`go_e4_cross_hardware=true`。
- 环境：2；代表 Kernel smoke：2/2。
- 正确性：两台机器均重放两个冻结工作负载、14 个函数体和两个违约布局。
- 统计边界：仅逐机器报告区间方向；区间跨 1 时结论为 uncertain，不把单机方向外推为普遍收益。

| 机器 | GPU | Guard p50 ms | Guard p95 ms | TritonPact/upstream 中位比 | 95% CI | 方向 |
| --- | --- | ---: | ---: | ---: | --- | --- |
| lab-rtx4090 | NVIDIA GeForce RTX 4090 | 0.065659 | 0.075382 | 2.6290 | [2.5800, 2.7104] | higher |
| lab-rtx5080 | NVIDIA GeForce RTX 5080 | 0.056137 | 0.065204 | 2.7443 | [2.6624, 2.8493] | higher |

完整环境、GPU 状态、交错顺序、原始样本、正确性明细和分项保存在同名 JSON。
