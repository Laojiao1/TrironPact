# e2 Access IR 覆盖

- 模式：`development_and_frozen_holdout`。
- 指纹：`3d8758b5db0221c5a282f6bb22ebb6829a8301abe4d141a112d88bd02986ad20`。
- 覆盖：{'total': 28, 'Supported': 27, 'Unknown': 1}。
- 边界：Supported 仅表示绑定后的全部访存点进入统一 IR；候选、Guard 和 Fast 资格另行核验。

| Kernel | 来源 | 类别 | 划分 | 状态 | load/store | 原因 |
| --- | --- | --- | --- | --- | --- | --- |
| triton_add | triton | elementwise_mapping | development | Supported | 2/1 | 全部绑定后访存访问点已归一化；不证明 Kernel 数值语义 |
| triton_where_zero | triton | elementwise_mapping | development | Supported | 1/1 | 全部绑定后访存访问点已归一化；不证明 Kernel 数值语义 |
| triton_interleave | triton | elementwise_mapping | holdout | Supported | 0/1 | 全部绑定后访存访问点已归一化；不证明 Kernel 数值语义 |
| triton_ravel | triton | elementwise_mapping | development | Supported | 0/1 | 全部绑定后访存访问点已归一化；不证明 Kernel 数值语义 |
| pytorch_sub | pytorch | elementwise_mapping | development | Supported | 2/1 | 全部绑定后访存访问点已归一化；不证明 Kernel 数值语义 |
| pytorch_double | pytorch | elementwise_mapping | development | Supported | 1/1 | 全部绑定后访存访问点已归一化；不证明 Kernel 数值语义 |
| pytorch_double_strided | pytorch | elementwise_mapping | holdout | Supported | 1/1 | 全部绑定后访存访问点已归一化；不证明 Kernel 数值语义 |
| triton_broadcast | triton | feature_broadcast | development | Supported | 2/1 | 全部绑定后访存访问点已归一化；不证明 Kernel 数值语义 |
| triton_where_broadcast | triton | feature_broadcast | development | Supported | 2/1 | 全部绑定后访存访问点已归一化；不证明 Kernel 数值语义 |
| pytorch_masked_add | pytorch | feature_broadcast | development | Supported | 3/1 | 全部绑定后访存访问点已归一化；不证明 Kernel 数值语义 |
| liger_swiglu | liger | feature_broadcast | holdout | Supported | 2/1 | 全部绑定后访存访问点已归一化；不证明 Kernel 数值语义 |
| liger_swiglu_tiled | liger | feature_broadcast | development | Supported | 2/1 | 全部绑定后访存访问点已归一化；不证明 Kernel 数值语义 |
| liger_geglu | liger | feature_broadcast | development | Supported | 2/1 | 全部绑定后访存访问点已归一化；不证明 Kernel 数值语义 |
| liger_fused_swiglu | liger | feature_broadcast | development | Supported | 2/1 | 全部绑定后访存访问点已归一化；不证明 Kernel 数值语义 |
| flag_slice | flag_gems | layout_2d | holdout | Unknown | 0/0 | 地址 tl.where 条件不是已绑定 constexpr |
| triton_permute | triton | layout_2d | development | Supported | 1/1 | 全部绑定后访存访问点已归一化；不证明 Kernel 数值语义 |
| triton_trans2d | triton | layout_2d | development | Supported | 1/1 | 全部绑定后访存访问点已归一化；不证明 Kernel 数值语义 |
| triton_flip | triton | layout_2d | development | Supported | 1/1 | 全部绑定后访存访问点已归一化；不证明 Kernel 数值语义 |
| triton_pair_flip | triton | layout_2d | holdout | Supported | 1/1 | 全部绑定后访存访问点已归一化；不证明 Kernel 数值语义 |
| triton_split | triton | layout_2d | development | Supported | 1/2 | 全部绑定后访存访问点已归一化；不证明 Kernel 数值语义 |
| triton_softmax | triton | row_reduction | development | Supported | 1/1 | 全部绑定后访存访问点已归一化；不证明 Kernel 数值语义 |
| liger_softmax | liger | row_reduction | development | Supported | 1/1 | 全部绑定后访存访问点已归一化；不证明 Kernel 数值语义 |
| liger_layernorm | liger | row_reduction | holdout | Supported | 3/3 | 全部绑定后访存访问点已归一化；不证明 Kernel 数值语义 |
| unsloth_layernorm | unsloth | row_reduction | development | Supported | 3/3 | 全部绑定后访存访问点已归一化；不证明 Kernel 数值语义 |
| liger_rmsnorm | liger | row_reduction | development | Supported | 2/2 | 全部绑定后访存访问点已归一化；不证明 Kernel 数值语义 |
| liger_block_rmsnorm | liger | row_reduction | development | Supported | 2/2 | 全部绑定后访存访问点已归一化；不证明 Kernel 数值语义 |
| unsloth_rmsnorm | unsloth | row_reduction | holdout | Supported | 2/2 | 全部绑定后访存访问点已归一化；不证明 Kernel 数值语义 |
| unsloth_gemma_rmsnorm | unsloth | row_reduction | development | Supported | 2/2 | 全部绑定后访存访问点已归一化；不证明 Kernel 数值语义 |
