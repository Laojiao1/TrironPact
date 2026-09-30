# e1 能力缺口

开发集语法频次（不代表规则已实现）：

- broadcast：7 个 Kernel。
- cast：7 个 Kernel。
- reduction：6 个 Kernel。
- pointer_update：5 个 Kernel。
- layout_transform：4 个 Kernel。
- branch：3 个 Kernel。

| Kernel | 来源 | 类别 | 集合 | AST | load/store | grid 轴 | mask | 首个拒绝原因 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| triton_add | triton | elementwise_mapping | development | Supported | 2/1 | [0] | 3 | 访存语法已归一化；尚未核对独立算子语义 |
| triton_where_zero | triton | elementwise_mapping | development | Unsupported | 1/1 | [] | 0 | 不支持的索引表达式：tl.arange(0, BLOCK_SIZE)[None, :] |
| triton_interleave | triton | elementwise_mapping | holdout | Unsupported | 0/1 | [] | 0 | 缺少显式 load/store 访问 |
| triton_ravel | triton | elementwise_mapping | development | Unsupported | 0/1 | [] | 0 | 第 4 行重复定义变量或覆盖参数 |
| pytorch_sub | pytorch | elementwise_mapping | development | Unknown | 2/1 | [0] | 3 | arange 块大小缺少 constexpr 声明 |
| pytorch_double | pytorch | elementwise_mapping | development | Unknown | 1/1 | [0] | 2 | arange 块大小缺少 constexpr 声明 |
| pytorch_double_strided | pytorch | elementwise_mapping | holdout | Unsupported | 1/1 | [0, 1] | 0 | 不支持的索引表达式：y_offsets[:, None] |
| triton_broadcast | triton | feature_broadcast | development | Unsupported | 2/1 | [] | 0 | 不支持第 7 行的控制流或语句 |
| triton_where_broadcast | triton | feature_broadcast | development | Unsupported | 2/1 | [] | 0 | 不支持的索引表达式：tl.arange(0, BLOCK_SIZE)[None, :] |
| pytorch_masked_add | pytorch | feature_broadcast | development | Unsupported | 3/1 | [0] | 4 | 访存调用存在未建模的嵌套形式 |
| liger_swiglu | liger | feature_broadcast | holdout | Unsupported | 2/1 | [0] | 3 | 不支持第 8 行的控制流或语句 |
| liger_swiglu_tiled | liger | feature_broadcast | development | Unsupported | 2/1 | [0, 1] | 3 | 不支持第 8 行的控制流或语句 |
| liger_geglu | liger | feature_broadcast | development | Unsupported | 2/1 | [0] | 3 | 不支持第 6 行的控制流或语句 |
| liger_fused_swiglu | liger | feature_broadcast | development | Unsupported | 2/1 | [0] | 3 | 不支持第 7 行的控制流或语句 |
| flag_slice | flag_gems | layout_2d | holdout | Unsupported | 1/1 | [0, 1] | 2 | 不支持的索引表达式：tl.where(dim == 0, start + out_idx_0 * step, out_idx_0) |
| triton_permute | triton | layout_2d | development | Unsupported | 1/1 | [] | 0 | 缺少显式 load/store 访问 |
| triton_trans2d | triton | layout_2d | development | Unsupported | 1/1 | [] | 0 | 缺少显式 load/store 访问 |
| triton_flip | triton | layout_2d | development | Unsupported | 1/1 | [] | 0 | 第 8 行重复定义变量或覆盖参数 |
| triton_pair_flip | triton | layout_2d | holdout | Unsupported | 1/1 | [0] | 0 | 不支持第 5 行的控制流或语句 |
| triton_split | triton | layout_2d | development | Unsupported | 1/2 | [] | 0 | 不支持第 6 行的控制流或语句 |
| triton_softmax | triton | row_reduction | development | Unsupported | 1/1 | [] | 0 | 不支持第 6 行的控制流或语句 |
| liger_softmax | liger | row_reduction | development | Supported | 1/1 | [0] | 2 | 访存语法已归一化；尚未核对独立算子语义 |
| liger_layernorm | liger | row_reduction | holdout | Unknown | 3/3 | [0] | 4 | 访存基址缺少唯一指针绑定 |
| unsloth_layernorm | unsloth | row_reduction | development | Unsupported | 3/3 | [0] | 4 | 不支持第 19 行的控制流或语句 |
| liger_rmsnorm | liger | row_reduction | development | Unsupported | 2/2 | [0] | 3 | 不支持第 37 行的控制流或语句 |
| liger_block_rmsnorm | liger | row_reduction | development | Unsupported | 2/2 | [0] | 3 | 不支持第 40 行的控制流或语句 |
| unsloth_rmsnorm | unsloth | row_reduction | holdout | Unsupported | 2/2 | [0] | 3 | 不支持第 24 行的控制流或语句 |
| unsloth_gemma_rmsnorm | unsloth | row_reduction | development | Unsupported | 2/2 | [0] | 3 | 不支持第 21 行的控制流或语句 |
| triton_signed_neg | triton | elementwise_mapping | challenge | Unsupported | 1/1 | [] | 0 | tl.load/store 缺少显式 mask |
| triton_cat | triton | layout_2d | challenge | Unsupported | 2/1 | [] | 0 | tl.load/store 缺少显式 mask |
| triton_tutorial_softmax | triton | row_reduction | challenge | Unsupported | 1/1 | [0] | 2 | 不支持第 7 行的控制流或语句 |
| triton_tutorial_layernorm | triton | row_reduction | challenge | Unsupported | 5/3 | [0] | 6 | 不支持第 16 行的控制流或语句 |
| triton_gather | triton | indirect_index | challenge | Unsupported | 2/1 | [] | 0 | 不支持的索引表达式：tl.arange(0, src_dim0)[:, None] |
| triton_trans4d | triton | tensor_descriptor | challenge | Unsupported | 0/0 | [] | 0 | 不支持第 19 行的控制流或语句 |
| triton_reshape_template | triton | source_template | challenge | Unknown | 1/1 | [] | 0 | 访存基址缺少唯一指针绑定 |
| liger_swiglu_backward | liger | inplace_multi_store | challenge | Unsupported | 3/2 | [0] | 5 | 不支持第 8 行的控制流或语句 |
| unsloth_swiglu | unsloth | feature_broadcast | challenge | Unsupported | 2/1 | [0] | 3 | 不支持第 4 行的控制流或语句 |
| unsloth_rope | unsloth | inplace_rope | challenge | Unsupported | 4/2 | [0, 1] | 6 | 不支持第 37 行的控制流或语句 |
| flag_cat | flag_gems | layout_2d | challenge | Unsupported | 1/1 | [0, 1] | 2 | 不支持第 27 行的控制流或语句 |
| flag_complex_transpose | flag_gems | complex_static_loop | challenge | Unsupported | 1/1 | [0] | 0 | 不支持第 25 行的控制流或语句 |

摸底结果归档但不用于规则设计排序；e2 只读取 development 的需求，留出集正式恢复评估在规则冻结后进行
