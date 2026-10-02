# e3 开发与证据记录

基线为 `dd0dd74 phase6 e2`，分支 `eval`。本阶段只实现真实边界变异、公平对照、L1/L2/L3 风险分层、真实候选 SMT 与精化审计；没有接入在线 Fast，也没有运行 e4 跨硬件正式实验。

| 工作块 | 命令 | 结果 | 证据 |
| --- | --- | --- | --- |
| 风险规格与失败闭合 | `python -m pytest -q test/test_e3_risk.py` | 退出码 0；2 passed | `test/test_e3_risk.py` |
| L1/L2/L3 独立 worker | `python -m bench.e3.risk --seed 17 --timeout 180` | 退出码 0；4/4 完成，4 L1、3 L2、1 L3 | `results/e3_real_risk_cases.md/.json` |
| 公平变异对照 | `python -m pytest -q test/test_e3_mutation.py test/test_e3_risk.py`；`python -m bench.e3.mutations` | 退出码 0；5 passed；共享域、种子和预算检查通过 | `results/e3_mutation_comparison.md/.json` |
| 真实候选 SMT | `python -m pytest -q test/test_e3_smt.py`；`python -m bench.e3.smt --timeout-ms 2000` | 退出码 0；158 个候选，1 个真实等价重复，0 冲突；三维域保守 Unknown | `results/e3_smt_audit.md/.json` |
| 精化与撤销 | `python -m bench.e3.refinement` | 退出码 0；0 个 Guard=True 反例、0 修订、0 Fast 撤销 | `results/e3_refinement_audit.md/.json` |
| 阶段完整回归 | `python -m bench.e3.validate` | 退出码 0；134 passed；旧隔离 11/11；`go_e3=true` | `results/e3_regression.md/.json`、`e3_regression_isolation.md/.json` |

公开 L3 状态在 2026-10-02 核对：Unsloth RMSNorm PR #10617 已合并；LayerNorm PR #10675 仍为 Open。本地正式 L3 条目使用与冻结 `unsloth_layernorm` 语义直接对应的 #10675。Triton add 和 Liger softmax 的本地复现只形成 L1/L2，不借用无关 Issue 冒充上游漏洞。

