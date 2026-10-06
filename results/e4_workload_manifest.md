# e4a 工作负载与接入清单

- 状态：`go`；计数：`{'workloads': 2, 'integration_steps': 14, 'distinct_kernels': 14, 'by_source': {'pytorch': 4, 'triton': 1, 'liger': 7, 'unsloth': 2}, 'holdout_kernels': 4}`。
- 边界：函数体按 function_sha256 去重；留出条目只执行冻结评估，不反馈修改 e2 规则。

| 工作负载 | 调用位置 | Kernel | 来源 | 固定提交 | 集合 | 状态 |
| --- | --- | --- | --- | --- | --- | --- |
| inductor_custom_kernel_graph | pointwise.sub | `pytorch_sub` | pytorch | `a6b28b689562679531a842f27d4e39ba50bb34f1` | development | traceable |
| inductor_custom_kernel_graph | pointwise.double | `pytorch_double` | pytorch | `a6b28b689562679531a842f27d4e39ba50bb34f1` | development | traceable |
| inductor_custom_kernel_graph | pointwise.masked_add | `pytorch_masked_add` | pytorch | `a6b28b689562679531a842f27d4e39ba50bb34f1` | development | traceable |
| inductor_custom_kernel_graph | pointwise.add | `triton_add` | triton | `f394c9bb86b2eaca08b0e11c4cd397d19a56c693` | development | traceable |
| inductor_custom_kernel_graph | layout.double_strided | `pytorch_double_strided` | pytorch | `a6b28b689562679531a842f27d4e39ba50bb34f1` | holdout | traceable |
| transformer_layer_trace | norm.liger_rmsnorm | `liger_rmsnorm` | liger | `40a9d8a61bdc546f4e70220619d4db9760fa80b1` | development | traceable |
| transformer_layer_trace | norm.unsloth_rmsnorm | `unsloth_rmsnorm` | unsloth | `7ebf192674a217faa15891de3e95a517fb5d890e` | holdout | traceable |
| transformer_layer_trace | norm.liger_layernorm | `liger_layernorm` | liger | `40a9d8a61bdc546f4e70220619d4db9760fa80b1` | holdout | traceable |
| transformer_layer_trace | norm.unsloth_layernorm | `unsloth_layernorm` | unsloth | `7ebf192674a217faa15891de3e95a517fb5d890e` | development | traceable |
| transformer_layer_trace | activation.swiglu | `liger_swiglu` | liger | `40a9d8a61bdc546f4e70220619d4db9760fa80b1` | holdout | traceable |
| transformer_layer_trace | activation.swiglu_tiled | `liger_swiglu_tiled` | liger | `40a9d8a61bdc546f4e70220619d4db9760fa80b1` | development | traceable |
| transformer_layer_trace | activation.geglu | `liger_geglu` | liger | `40a9d8a61bdc546f4e70220619d4db9760fa80b1` | development | traceable |
| transformer_layer_trace | activation.fused_swiglu | `liger_fused_swiglu` | liger | `40a9d8a61bdc546f4e70220619d4db9760fa80b1` | development | traceable |
| transformer_layer_trace | activation.softmax | `liger_softmax` | liger | `40a9d8a61bdc546f4e70220619d4db9760fa80b1` | development | traceable |

## 核验

- [x] two_workloads
- [x] ten_distinct_function_bodies
- [x] all_steps_traceable
- [x] all_guards_frozen_supported
- [x] holdout_not_used_for_rule_changes
