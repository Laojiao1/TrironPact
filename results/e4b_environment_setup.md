# e4b 原生 Linux 环境配置记录

- 状态：`validation_complete`；`go_e4_cross_hardware=true`。
- 代码基线：两台机器均为 `eval@fcac52b0f48520e9a7c3e900cd2fa61d5d2c3e6c`；正式运行绑定相同 e4b 源码指纹 `c3458cb660febd1d96451a74014f05801deb550e895369a3cd254bf8d8b19146`。
- 依赖：同一个 Python 3.12 wheelhouse 离线安装；归档 SHA-256 为 `a098aa40a557698e928a1f343f2272f8d1d2d3ef0e06a37f47338ddef3046906`，两机逐 wheel 校验通过。
- 边界：代表 Kernel smoke、冻结工作负载正确性和正式计时均已完成；有限输入结果不证明域外布局安全或普遍性能收益。

| 环境 | 系统 / CPU | GPU / 驱动 | Python / Torch / Triton | 状态 |
| --- | --- | --- | --- | --- |
| `lab_rtx4090_ada` | Ubuntu 24.04.4；i9-13900K；123 GiB RAM | RTX 4090 24564 MiB；595.99.02 | 3.12.3 / 2.11.0+cu128 / 3.6.0 | smoke 与正式复核通过 |
| `lab_rtx5080_blackwell` | Ubuntu 24.04.4；Ryzen 7 9800X3D；30 GiB RAM | RTX 5080 16303 MiB；595.84 | 3.12.3 / 2.11.0+cu128 / 3.6.0 | 补齐 python3.12-dev 后 smoke 与正式复核通过 |

两台机器均使用 `/home/d514/work/venvs/tritonpact-e4b`，系统未安装 `nvcc`；PyTorch wheel 报告 CUDA 12.8。导入 PyTorch 时仍有 NumPy 未安装警告，但冻结协议未调用 NumPy，smoke、正确性和计时均成功，因此没有为消除非功能警告扩展 Python 依赖。5080 最初因缺少 `Python.h` 在 Kernel 启动前失败，安装与系统 Python 匹配的 `python3.12-dev 3.12.3-1ubuntu0.17` 后按原协议通过；该失败未计作 Kernel 结果。

正式计时窗口为 2026-10-12 09:12～09:17（Asia/Shanghai）。计时前两块 GPU 均为 0% 利用率；运行器保存测量前后温度、SM 时钟、功耗和 P-state。实验室共享主机没有独占调度证明，这一限制保留在证据边界中。

当前 `obtained_environments=2`、`configured_environments=2`、`native_smoke_passed=2`、`formal_experiments_run=2`。两机均完成两个工作负载、14 个函数体和两个违约布局的正确性复核，已知误放行仍为 0。完整 TritonPact 分派相对 upstream 在这两个极小图上均表现为稳定额外开销；不宣称加速，具体原始样本和区间见 `e4_cross_hardware_performance.json`。

原始证据：

- `results/e4b_environment_4090_raw.txt`
- `results/e4b_environment_5080_raw.txt`
- `results/e4b_offline_install_4090.log`
- `results/e4b_offline_install_5080.log`
- `results/e4b_environment_setup.json`
- `results/e4b_machine_4090.json`
- `results/e4b_machine_5080.json`
- `results/e4_cross_hardware_performance.json`
