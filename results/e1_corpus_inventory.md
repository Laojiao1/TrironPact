# e1 真实语料清单

{
  "registered": 40,
  "external": 40,
  "sources": 5,
  "syntax": {
    "Supported": 2,
    "Unsupported": 34,
    "Unknown": 4
  },
  "positive_declared": 28,
  "semantic_complete": 28,
  "positive_by_family": {
    "elementwise_mapping": 7,
    "feature_broadcast": 7,
    "layout_2d": 6,
    "row_reduction": 8
  },
  "splits": {
    "development": 21,
    "holdout": 7,
    "challenge": 12
  },
  "candidate_complete": 0,
  "guard_available": 0,
  "fast_feasible": 0,
  "integrated": 0
}

| ID | 来源 | 类别 | 集合 | AST | 语义 |
| --- | --- | --- | --- | --- | --- |
| triton_add | triton | elementwise_mapping | development | Supported | complete_in_smoke_domain |
| triton_where_zero | triton | elementwise_mapping | development | Unsupported | complete_in_smoke_domain |
| triton_interleave | triton | elementwise_mapping | holdout | Unsupported | complete_in_smoke_domain |
| triton_ravel | triton | elementwise_mapping | development | Unsupported | complete_in_smoke_domain |
| pytorch_sub | pytorch | elementwise_mapping | development | Unknown | complete_in_smoke_domain |
| pytorch_double | pytorch | elementwise_mapping | development | Unknown | complete_in_smoke_domain |
| pytorch_double_strided | pytorch | elementwise_mapping | holdout | Unsupported | complete_in_smoke_domain |
| triton_broadcast | triton | feature_broadcast | development | Unsupported | complete_in_smoke_domain |
| triton_where_broadcast | triton | feature_broadcast | development | Unsupported | complete_in_smoke_domain |
| pytorch_masked_add | pytorch | feature_broadcast | development | Unsupported | complete_in_smoke_domain |
| liger_swiglu | liger | feature_broadcast | holdout | Unsupported | complete_in_smoke_domain |
| liger_swiglu_tiled | liger | feature_broadcast | development | Unsupported | complete_in_smoke_domain |
| liger_geglu | liger | feature_broadcast | development | Unsupported | complete_in_smoke_domain |
| liger_fused_swiglu | liger | feature_broadcast | development | Unsupported | complete_in_smoke_domain |
| flag_slice | flag_gems | layout_2d | holdout | Unsupported | complete_in_smoke_domain |
| triton_permute | triton | layout_2d | development | Unsupported | complete_in_smoke_domain |
| triton_trans2d | triton | layout_2d | development | Unsupported | complete_in_smoke_domain |
| triton_flip | triton | layout_2d | development | Unsupported | complete_in_smoke_domain |
| triton_pair_flip | triton | layout_2d | holdout | Unsupported | complete_in_smoke_domain |
| triton_split | triton | layout_2d | development | Unsupported | complete_in_smoke_domain |
| triton_softmax | triton | row_reduction | development | Unsupported | complete_in_smoke_domain |
| liger_softmax | liger | row_reduction | development | Supported | complete_in_smoke_domain |
| liger_layernorm | liger | row_reduction | holdout | Unknown | complete_in_smoke_domain |
| unsloth_layernorm | unsloth | row_reduction | development | Unsupported | complete_in_smoke_domain |
| liger_rmsnorm | liger | row_reduction | development | Unsupported | complete_in_smoke_domain |
| liger_block_rmsnorm | liger | row_reduction | development | Unsupported | complete_in_smoke_domain |
| unsloth_rmsnorm | unsloth | row_reduction | holdout | Unsupported | complete_in_smoke_domain |
| unsloth_gemma_rmsnorm | unsloth | row_reduction | development | Unsupported | complete_in_smoke_domain |
| triton_signed_neg | triton | elementwise_mapping | challenge | Unsupported | not_required_challenge |
| triton_cat | triton | layout_2d | challenge | Unsupported | not_required_challenge |
| triton_tutorial_softmax | triton | row_reduction | challenge | Unsupported | not_required_challenge |
| triton_tutorial_layernorm | triton | row_reduction | challenge | Unsupported | not_required_challenge |
| triton_gather | triton | indirect_index | challenge | Unsupported | not_required_challenge |
| triton_trans4d | triton | tensor_descriptor | challenge | Unsupported | not_required_challenge |
| triton_reshape_template | triton | source_template | challenge | Unknown | not_required_challenge |
| liger_swiglu_backward | liger | inplace_multi_store | challenge | Unsupported | not_required_challenge |
| unsloth_swiglu | unsloth | feature_broadcast | challenge | Unsupported | not_required_challenge |
| unsloth_rope | unsloth | inplace_rope | challenge | Unsupported | not_required_challenge |
| flag_cat | flag_gems | layout_2d | challenge | Unsupported | not_required_challenge |
| flag_complex_transpose | flag_gems | complex_static_loop | challenge | Unsupported | not_required_challenge |

仅 e1 语料与有界语义证据。语法、语义完整、候选、Guard、Fast、接入分母分列；旧15个历史样本另存，未混入这40个真实样本。
