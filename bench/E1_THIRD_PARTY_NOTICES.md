# e1 第三方语料说明

e1 只保存固定提交中的必要 Kernel、官方测试/参考与许可证副本，用于研究审计和可重复实验。原始函数体与原注释保持不变；本地生成的加载封套只补充导入、固定 helper 常量和外层作用域去缩进，差异与哈希见 `bench/e1/catalog.json`。

| 来源 | 固定提交 | 许可与附加说明 |
| --- | --- | --- |
| triton-lang/triton | `f394c9bb86b2eaca08b0e11c4cd397d19a56c693` | MIT；完整文本保存在 `bench/e1/upstream/triton/LICENSE`。 |
| linkedin/Liger-Kernel | `40a9d8a61bdc546f4e70220619d4db9760fa80b1` | BSD-2-Clause；完整文本保存在 `bench/e1/upstream/liger/LICENSE`。 |
| unslothai/unsloth | `7ebf192674a217faa15891de3e95a517fb5d890e` | 仓库根许可为 Apache-2.0；保存的两个测试文件另带 `AGPL-3.0-only` SPDX，原文件头未修改。完整根许可保存在 `bench/e1/upstream/unsloth/LICENSE`。 |
| FlagOpen/FlagGems | `b2044acafb7f72a19312158edf0d2bf333e26891` | Apache-2.0；完整文本保存在 `bench/e1/upstream/flag_gems/LICENSE`。 |
| pytorch/pytorch | `a6b28b689562679531a842f27d4e39ba50bb34f1` | PyTorch/Caffe2 BSD-style terms；完整文本保存在 `bench/e1/upstream/pytorch/LICENSE`。 |

上游版权归各自权利人。本清单不把项目名、单次运行或 AST 可解析性当作独立算法来源、一般安全证明或 Fast 放行依据。
