# e1 最近邻工作与基线冻结

- 核对日期：2026-09-30。
- 主要投稿方向：CGO-style CCF B compiler and program-analysis track。
- 外部可执行基线：triton-verify，状态 `passed`。

| 工作 | 正式状态 | 代码/基线 | 与 TritonPact 的边界 |
| --- | --- | --- | --- |
| M2K: Making the Model-Kernel Interface Explicit for Reliable CUDA Kernel Verification | SOSP 2026 accepted paper | qualitative_only | 目标为 CUDA/LLM 系统和符号执行；不直接恢复 Triton Python wrapper 与物理 layout 的在线分派契约。 |
| Triton-Sanitizer / TileLens | Triton-Sanitizer, ASPLOS 2026; code evolved into TileLens | frozen_public_code | 以给定输入/trace 的 sanitizer 为主，不等价于 wrapper 物理 layout 前置条件恢复或在线 Fast 资格。 |
| triton-verify | unpublished public prototype; upstream RFC opened 2026-06-07 | executable_smoke | buffer size 与 stride 需手工提供，只检查 pre-optimization TTIR 指针安全，不检查数值语义。 |
| Towards More Complete Constraints for Deep Learning Library Testing via Complementary Set Guided Refinement | ISSTA 2024 | qualitative_only | 对象是框架 API/图测试约束，不建模 Triton kernel 物理地址、wrapper stride/offset 或分派。 |
| CuSafe: Capturing Memory Corruption on NVIDIA GPUs | USENIX Security 2026 | qualitative_only | 运行时 CUDA sanitizer，不恢复 Triton wrapper 契约；官方论文页未给出可固定代码仓库。 |

详细固定提交、复现命令、stdout/stderr 和证据边界见同名 JSON。无法运行的工作只作定性比较。
