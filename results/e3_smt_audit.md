# e3 真实候选 SMT 审计

- 生成时间：2026-10-02T13:18:03.057010+08:00。
- 状态：`go`；计数：`{'kernels': 28, 'audited': 27, 'smt_encoded': 26, 'real_candidates': 158, 'real_candidate_deletions': 1, 'kernels_with_conflict': 0, 'unknown_queries': 25, 'synthetic_deletions': 0}`。
- 贡献定位：真实候选存在可删除的同域等价重复。
- 合成 fixture 删除不计入真实收益；未编码的 AccessSpan 和求解 Unknown 全部保留。

| Kernel | 状态 | 候选 | 真实删除 | 等价组 | 查询状态 |
| --- | --- | ---: | --- | --- | --- |
| `triton_add` | Supported | 6 | [] | [] | {'sat': 5, 'unknown': 1} |
| `triton_where_zero` | Supported | 4 | [] | [] | {'sat': 4, 'unknown': 1} |
| `triton_interleave` | Supported | 2 | [] | [] | {'sat': 1} |
| `triton_ravel` | Supported | 2 | [] | [] | {'sat': 1} |
| `pytorch_sub` | Supported | 6 | [] | [] | {'sat': 5, 'unknown': 1} |
| `pytorch_double` | Supported | 4 | [] | [] | {'sat': 4, 'unknown': 1} |
| `pytorch_double_strided` | Supported | 4 | [] | [] | {'sat': 4, 'unknown': 1} |
| `triton_broadcast` | Supported | 6 | [] | [] | {'sat': 5, 'unknown': 1} |
| `triton_where_broadcast` | Supported | 6 | [] | [] | {'sat': 5, 'unknown': 1} |
| `pytorch_masked_add` | Supported | 8 | [] | [] | {'sat': 6, 'unknown': 1} |
| `liger_swiglu` | Supported | 6 | [] | [] | {'sat': 5, 'unknown': 1} |
| `liger_swiglu_tiled` | Supported | 6 | [] | [] | {'sat': 5, 'unknown': 1} |
| `liger_geglu` | Supported | 6 | [] | [] | {'sat': 5, 'unknown': 1} |
| `liger_fused_swiglu` | Supported | 6 | [3] | [[1, 3]] | {'sat': 4, 'unknown': 1, 'unsat': 1} |
| `flag_slice` | Unknown | 0 | [] | [] | {} |
| `triton_permute` | Supported | 4 | [] | [] | {'sat': 4, 'unknown': 1} |
| `triton_trans2d` | Supported | 4 | [] | [] | {'sat': 4, 'unknown': 1} |
| `triton_flip` | Unknown | 4 | [] | [] | {'unknown': 1} |
| `triton_pair_flip` | Supported | 4 | [] | [] | {'sat': 4, 'unknown': 1} |
| `triton_split` | Supported | 6 | [] | [] | {'sat': 5, 'unknown': 1} |
| `triton_softmax` | Supported | 4 | [] | [] | {'sat': 4, 'unknown': 1} |
| `liger_softmax` | Supported | 4 | [] | [] | {'sat': 4, 'unknown': 1} |
| `liger_layernorm` | Supported | 12 | [] | [] | {'sat': 8, 'unknown': 1} |
| `unsloth_layernorm` | Supported | 12 | [] | [] | {'sat': 8, 'unknown': 1} |
| `liger_rmsnorm` | Supported | 8 | [] | [] | {'sat': 6, 'unknown': 1} |
| `liger_block_rmsnorm` | Supported | 8 | [] | [] | {'sat': 6, 'unknown': 1} |
| `unsloth_rmsnorm` | Supported | 8 | [] | [] | {'sat': 6, 'unknown': 1} |
| `unsloth_gemma_rmsnorm` | Supported | 8 | [] | [] | {'sat': 6, 'unknown': 1} |
