# e2 Guard 与 Fast 可行覆盖

- 模式：`development_and_frozen_holdout`。
- 计数：{'total': 28, 'guard_available': 27, 'fast_feasible': 27}。
- Guard 类别：{'elementwise_mapping': 7, 'feature_broadcast': 7, 'layout_2d': 5, 'row_reduction': 8}。
- Fast 类别：{'elementwise_mapping': 7, 'feature_broadcast': 7, 'layout_2d': 5, 'row_reduction': 8}。
- 边界：Fast 可行仅表示冻结输入上 Guard 放行且数值参考通过；不授予在线 Fast，也不证明一般安全。

| Kernel | 来源 | 类别 | Guard | Fast 可行 | 当前 smoke |
| --- | --- | --- | --- | --- | --- |
| triton_add | triton | elementwise_mapping | Supported | True | True |
| triton_where_zero | triton | elementwise_mapping | Supported | True | True |
| triton_interleave | triton | elementwise_mapping | Supported | True | True |
| triton_ravel | triton | elementwise_mapping | Supported | True | True |
| pytorch_sub | pytorch | elementwise_mapping | Supported | True | True |
| pytorch_double | pytorch | elementwise_mapping | Supported | True | True |
| pytorch_double_strided | pytorch | elementwise_mapping | Supported | True | True |
| triton_broadcast | triton | feature_broadcast | Supported | True | True |
| triton_where_broadcast | triton | feature_broadcast | Supported | True | True |
| pytorch_masked_add | pytorch | feature_broadcast | Supported | True | True |
| liger_swiglu | liger | feature_broadcast | Supported | True | True |
| liger_swiglu_tiled | liger | feature_broadcast | Supported | True | True |
| liger_geglu | liger | feature_broadcast | Supported | True | True |
| liger_fused_swiglu | liger | feature_broadcast | Supported | True | True |
| flag_slice | flag_gems | layout_2d | Unknown | False | True |
| triton_permute | triton | layout_2d | Supported | True | True |
| triton_trans2d | triton | layout_2d | Supported | True | True |
| triton_flip | triton | layout_2d | Supported | True | True |
| triton_pair_flip | triton | layout_2d | Supported | True | True |
| triton_split | triton | layout_2d | Supported | True | True |
| triton_softmax | triton | row_reduction | Supported | True | True |
| liger_softmax | liger | row_reduction | Supported | True | True |
| liger_layernorm | liger | row_reduction | Supported | True | True |
| unsloth_layernorm | unsloth | row_reduction | Supported | True | True |
| liger_rmsnorm | liger | row_reduction | Supported | True | True |
| liger_block_rmsnorm | liger | row_reduction | Supported | True | True |
| unsloth_rmsnorm | unsloth | row_reduction | Supported | True | True |
| unsloth_gemma_rmsnorm | unsloth | row_reduction | Supported | True | True |
