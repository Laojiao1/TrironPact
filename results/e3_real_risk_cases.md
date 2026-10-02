# e3 真实风险案例

- 生成时间：2026-10-02T13:13:24.781333+08:00。
- 状态：`go`；计数：`{'registered': 4, 'complete': 4, 'l1_witnesses': 4, 'l2_wrapper_defenses': 3, 'l3_public_items': 1, 'by_source': {'triton': 1, 'liger': 1, 'unsloth': 2}}`。
- L1、L2、L3 分层互不替代；WSL2 复制耗时仅作诊断，不属于 e4 性能结论。

| 案例 | 来源 | 原始调用 | Guard | Relayout | 复制字节 | L2 | L3 |
| --- | --- | --- | --- | --- | ---: | --- | --- |
| `triton_add_x_stride` | triton | numeric_mismatch | False | correct | 516 | 否 | 0 |
| `liger_softmax_x_stride` | liger | numeric_mismatch | False | correct | 32768 | 是 | 0 |
| `unsloth_layernorm_x_stride` | unsloth | numeric_mismatch | False | correct | 2560 | 是 | https://github.com/unslothai/unsloth/pull/10675 |
| `unsloth_layernorm_affine_stride` | unsloth | numeric_mismatch | False | correct | 640 | 是 | https://github.com/unslothai/unsloth/pull/10675 |

## 边界

- **L1**：底层 Kernel 对合法 PyTorch 物理布局的前置条件敏感性；没有接口承诺时不称为上游漏洞
- **L2**：固定源码中真实存在的 contiguous 防御、复制字节与 WSL2 诊断耗时
- **L3**：截至 2026-10-02 可公开核对的 Issue/PR；只登记与本地问题模式语义对应的项目
