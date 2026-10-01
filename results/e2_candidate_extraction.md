# e2 候选提取

- 模式：`development_and_frozen_holdout`。
- 计数：{'total': 28, 'candidate_complete': 27, 'candidate_incomplete': 1, 'candidates': 158}。
- 类别：{'elementwise_mapping': 7, 'feature_broadcast': 7, 'layout_2d': 5, 'row_reduction': 8}。
- 边界：候选只覆盖冻结 wrapper/grid 的物理映射与访存 span；reduction/普通算术数值语义未被静态证明。

| Kernel | 来源 | 类别 | 状态 | 访问点 | span |
| --- | --- | --- | --- | --- | --- |
| triton_add | triton | elementwise_mapping | complete | 3 | [[0, 128], [0, 128], [0, 128]] |
| triton_where_zero | triton | elementwise_mapping | complete | 2 | [[0, 1023], [0, 1023]] |
| triton_interleave | triton | elementwise_mapping | complete | 1 | [[0, 255]] |
| triton_ravel | triton | elementwise_mapping | complete | 1 | [[0, 255]] |
| pytorch_sub | pytorch | elementwise_mapping | complete | 3 | [[0, 128], [0, 128], [0, 128]] |
| pytorch_double | pytorch | elementwise_mapping | complete | 2 | [[0, 128], [0, 128]] |
| pytorch_double_strided | pytorch | elementwise_mapping | complete | 2 | [[0, 255], [0, 255]] |
| triton_broadcast | triton | feature_broadcast | complete | 3 | [[0, 2047], [0, 63], [0, 2047]] |
| triton_where_broadcast | triton | feature_broadcast | complete | 3 | [[0, 31], [0, 1023], [0, 1023]] |
| pytorch_masked_add | pytorch | feature_broadcast | complete | 4 | [[0, 128], [0, 128], [0, 128], [0, 128]] |
| liger_swiglu | liger | feature_broadcast | complete | 3 | [[0, 8191], [0, 8191], [0, 8191]] |
| liger_swiglu_tiled | liger | feature_broadcast | complete | 3 | [[0, 8191], [0, 8191], [0, 8191]] |
| liger_geglu | liger | feature_broadcast | complete | 3 | [[0, 8191], [0, 8191], [0, 8191]] |
| liger_fused_swiglu | liger | feature_broadcast | complete | 3 | [[0, 16127], [256, 16383], [0, 8191]] |
| flag_slice | flag_gems | layout_2d | Unknown | 0 | [] |
| triton_permute | triton | layout_2d | complete | 2 | [[0, 4095], [0, 4095]] |
| triton_trans2d | triton | layout_2d | complete | 2 | [[0, 255], [0, 255]] |
| triton_flip | triton | layout_2d | complete | 2 | [[0, 4095], [0, 4095]] |
| triton_pair_flip | triton | layout_2d | complete | 2 | [[0, 15], [0, 15]] |
| triton_split | triton | layout_2d | complete | 3 | [[0, 255], [0, 127], [0, 127]] |
| triton_softmax | triton | row_reduction | complete | 2 | [[0, 127], [0, 127]] |
| liger_softmax | liger | row_reduction | complete | 2 | [[0, 8191], [0, 8191]] |
| liger_layernorm | liger | row_reduction | complete | 6 | [[0, 79], [0, 79], [0, 639], [0, 7], [0, 7], [0, 639]] |
| unsloth_layernorm | unsloth | row_reduction | complete | 6 | [[0, 639], [0, 79], [0, 79], [0, 7], [0, 7], [0, 639]] |
| liger_rmsnorm | liger | row_reduction | complete | 4 | [[0, 639], [0, 79], [0, 7], [0, 639]] |
| liger_block_rmsnorm | liger | row_reduction | complete | 4 | [[0, 639], [0, 79], [0, 7], [0, 639]] |
| unsloth_rmsnorm | unsloth | row_reduction | complete | 4 | [[0, 639], [0, 79], [0, 7], [0, 639]] |
| unsloth_gemma_rmsnorm | unsloth | row_reduction | complete | 4 | [[0, 639], [0, 79], [0, 7], [0, 639]] |
