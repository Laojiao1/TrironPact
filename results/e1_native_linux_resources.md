# e1 原生 Linux GPU 资源准备

- 登记日期：2026-09-30。
- 策略：2026-09-30 用户决定暂缓实机项：e1 只冻结候选、目标版本、公开成本和脚本；不预订、不登录、不付费。项目开发完成后、正式跨硬件实验前重新打开，记录实际 driver、OS image、时段、账单并运行只读 smoke。

| 候选 | GPU/架构 | 系统与驱动 | 时段 | 成本 | 状态 |
| --- | --- | --- | --- | --- | --- |
| runpod_rtx4090_ada | NVIDIA RTX 4090 24 GB / Ada | native Ubuntu 22.04 x86_64；Unknown until provisioned | marketplace capacity not reserved; user to schedule | $0.74/GPU-hour | not_acquired；smoke=not_run |
| runpod_a100_pcie_ampere | NVIDIA A100 PCIe 80 GB / Ampere | native Ubuntu 22.04 x86_64；Unknown until provisioned | marketplace capacity not reserved; user to schedule | $1.59/GPU-hour | not_acquired；smoke=not_run |

候选、目标版本和环境/代表 Kernel smoke 脚本已冻结，满足 e1 的资源准备要求。根据 2026-09-30 用户决定，实际租赁与两机 smoke 延期到项目开发完成后、正式跨硬件实验前；当前 `native_resource_ready=false`，且未运行任何 e4 性能实验。
