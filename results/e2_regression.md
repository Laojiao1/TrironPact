# e2 阶段状态与验收记录

生成时间：2026-10-01T12:38:13.400532+08:00
状态：`go`；go_e2=True。

七层计数：{'registered': 40, 'positive': 28, 'supported': 27, 'unknown': 1, 'unsupported': 0, 'candidate_complete': 27, 'guard_available': 27, 'fast_feasible': 27, 'integrated': 0, 'challenge': 12}。
类别分布：{'elementwise_mapping': 7, 'feature_broadcast': 7, 'layout_2d': 5, 'row_reduction': 8}。
来源分布：{'triton': 12, 'pytorch': 4, 'liger': 8, 'unsloth': 3}。

## 核验项

| 核验 | 结果 |
| --- | --- |
| phase6_current_and_go | 通过 |
| e1_fingerprint_current | 通过 |
| regression_and_legacy_isolation | 通过 |
| development_rules_frozen_before_holdout | 通过 |
| holdout_not_used_for_rule_development | 通过 |
| full_positive_denominator_reported | 通过 |
| supported_threshold | 通过 |
| candidate_threshold | 通过 |
| guard_threshold | 通过 |
| fast_threshold | 通过 |
| family_distribution | 通过 |
| source_distribution | 通过 |
| all_fast_smoke_current | 通过 |
| unknown_preserved | 通过 |
| no_online_or_e3_changes | 通过 |

## 拒绝与边界

- `flag_slice`：Unknown；地址 tl.where 条件不是已绑定 constexpr。
- e2 Fast 可行是冻结调用域的离线证据，未注册在线分派；reduction 仅证明访问域，数值正确性来自有限独立参考 smoke。
- CCF A backlog：TTIR 辅助解析、复杂静态循环、组合 reduction 数值证明、图级 Guard。

## 回归

- `python test/run_tests.py --quick --output /mnt/f/Project/Paper/Code/TritonPact/results/e2_regression_isolation.md`：退出码 0；127 passed in 69.78s (0:01:09)；旧隔离 11/11。
