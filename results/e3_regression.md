# e3 阶段状态与验收记录

生成时间：2026-10-02T14:52:47.870071+08:00
状态：`go`；go_e3=True。
源码指纹：`3355bf7160f330772f04cb4ba4b20e8ffe9802e73fc89f0033b4ef26b7ed3fdc`。

计数：`{'mutation_probes': 14, 'mutation_trials': 20, 'registered': 4, 'complete': 4, 'l1_witnesses': 4, 'l2_wrapper_defenses': 3, 'l3_public_items': 1, 'by_source': {'triton': 1, 'liger': 1, 'unsloth': 2}, 'smt_real_candidates': 158, 'smt_real_deletions': 1, 'actual_revisions': 0, 'fast_revocations': 0}`。

## 核验项

| 核验 | 结果 |
| --- | --- |
| e1_fingerprint_current | 通过 |
| e2_go_and_rule_fingerprint_current | 通过 |
| full_regression_and_legacy_isolation | 通过 |
| mutation_protocol_frozen_and_fair | 通过 |
| real_risk_minimum | 通过 |
| risk_layers_separated | 通过 |
| smt_real_candidates_audited | 通过 |
| refutation_and_revision_closed | 通过 |
| no_e4_or_online_changes | 通过 |

## 结论边界

- **mutation**：谓词引导在多数基线的首次见证尝试数上更少
- **smt**：真实候选存在可删除的同域等价重复
- **risk**：4 个 L1、3 个 L2、1 个公开 L3 条目；L1 未冒充上游漏洞
- **performance**：复制耗时仅为 WSL2 诊断，未提前运行 e4 正式实验

## 回归

- `python test/run_tests.py --quick --output /mnt/f/Project/Paper/Code/TritonPact/results/e3_regression_isolation.md`：退出码 0；134 passed in 78.94s (0:01:18)；旧隔离 11/11。
