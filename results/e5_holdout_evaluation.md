# e5a 冻结留出结果只读审计

- 状态：`complete`；模式：`read_only_audit_of_e2_frozen_holdout_v1`。
- e5a 未重新执行留出 Kernel；本报告核对 e2 冻结前后顺序、规则指纹与三份正式结果的一致性。
- 计数：{'holdout': 7, 'supported': 6, 'unknown': 1, 'fast_feasible': 6, 'known_false_allows': 0}。
- 边界：仅审计 e2 已归档的一次性留出结果；e5a 没有重新运行、调规则或把留出集改称开发集。

| Kernel | Access IR | Candidate | Guard | Fast | smoke current |
| --- | --- | --- | --- | --- | --- |
| `flag_slice` | Unknown | Unknown | Unknown | 否 | 是 |
| `liger_layernorm` | Supported | complete | Supported | 是 | 是 |
| `liger_swiglu` | Supported | complete | Supported | 是 | 是 |
| `pytorch_double_strided` | Supported | complete | Supported | 是 | 是 |
| `triton_interleave` | Supported | complete | Supported | 是 | 是 |
| `triton_pair_flip` | Supported | complete | Supported | 是 | 是 |
| `unsloth_rmsnorm` | Supported | complete | Supported | 是 | 是 |

## 核验

- [x] freeze_precedes_holdout_and_marks_evaluated
- [x] rule_fingerprint_still_current
- [x] same_seven_holdout_ids_in_all_reports
- [x] reports_use_frozen_holdout_mode
- [x] holdout_not_used_for_rule_development
- [x] six_supported_one_unknown_preserved
- [x] supported_holdout_smoke_current
- [x] unknown_not_fast
