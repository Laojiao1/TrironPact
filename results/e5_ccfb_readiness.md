# e5 CCF B readiness

- 状态：`not_ready`；`go_e5_ccfb_readiness=false`。
- 这些是项目路线的内部论文证据门槛，不是 CCF 会议官方录用规则。

| 最终条件 | 结果 | 原因 |
| --- | --- | --- |
| `1_real_corpus_coverage_and_holdout_thresholds` | 通过 | 证据满足 |
| `2_reproducible_risk_evidence` | 通过 | 证据满足 |
| `3_zero_known_holdout_false_allows` | 通过 | 证据满足 |
| `4_two_workloads_and_two_hardware_environments` | 未满足 | e4b 尚无两套原生 Linux GPU、实际 smoke、正确性和开销复核。 |
| `5_fair_system_baselines` | 通过 | 证据满足 |
| `6_correctness_argument_traceable` | 通过 | 证据满足 |
| `7_independent_second_review` | 未满足 | 尚无第二审阅人对至少 25% 正式 Kernel 及全部风险案例的独立记录。 |
| `8_secondary_and_zero_results_disclosed` | 通过 | 证据满足 |
| `9_current_commit_environment_and_data_bound` | 未满足 | e5a 尚在未提交开发工作树；提交后需在目标提交重新生成最终证据清单。 |
| `10_clean_checkout_one_click_verification` | 未满足 | 最终条件要求提交后的干净 checkout 重跑；e5a 开发验收不能提前满足。 |
