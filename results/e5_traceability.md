# e5a 主张与证据追踪

- 状态：`complete`；9/9 项引用完整。
- 每项均显式列出证据级别与限制；路径存在只证明资产可定位，不替代其内部结论。

| ID | 主张 | 证据级别 | 代码 | 测试 | 报告/原始数据 | 限制 |
| --- | --- | --- | --- | --- | --- | --- |
| `C1_corpus_truth` | 40 个固定开源函数体具有去重、来源、许可、绑定和语义审计；28 个正向样例具有独立参考。 | `source_and_bounded_empirical` | bench/e1/catalog.py<br>bench/e1/audit.py<br>bench/e1/catalog.json | test/test_e1_corpus.py | results/e1_corpus_inventory.json<br>results/e1_semantic_audit.json<br>results/e1_semantic_smoke.json | 语义审计目前由单人完成；冻结输入 smoke 不构成一般语义证明。 |
| `C2_access_translation` | 受支持 AST 在 Access IR 中保留指针基址、元素偏移、mask、元素宽度和访问点来源。 | `static_supported_subset` | pact/e2_access.py<br>bench/e2/coverage.py<br>artifact/e5_correctness.md | test/test_e2_access_ir.py<br>test/test_e2_invariants.py<br>test/test_e5_correctness.py | results/e2_access_ir_coverage.json<br>bench/e1/catalog.json<br>bench/e2/freeze.json | 仅覆盖文档列出的无歧义 AST 子集；动态地址选择、一般循环和间接索引 fail closed。 |
| `C3_candidate_sufficiency` | 候选条件在声明语义、绑定和冻结适用域内足以保证逻辑访问与物理访问兼容。 | `static_template_plus_external_semantics` | pact/e2_contracts.py<br>bench/e2/contracts.py<br>bench/e2/plans.py<br>artifact/e5_correctness.md | test/test_e2_contracts.py<br>test/test_e2_invariants.py<br>test/test_e5_correctness.py | results/e2_candidate_extraction.json<br>results/e1_semantic_audit.json<br>bench/e2/freeze.json | 充分性依赖外部语义和完整 wrapper 绑定；reduction 只证明访问域，不证明一般数值规约。 |
| `C4_guard_fail_closed` | Guard 仅在全部义务为 True 时放行；False 或 Unknown 均不进入对应 Fast。 | `executable_invariant` | pact/e2_guard.py<br>pact/guard_plan.py<br>pact/dispatch.py<br>artifact/e5_correctness.md | test/test_e2_guard.py<br>test/test_guard_dispatch.py<br>test/test_e5_correctness.py | results/e2_guard_coverage.json<br>results/e4_workload_correctness.json<br>results/e2_guard_smoke.json | 零已知误放行只适用于冻结样例与声明域，不是任意 Triton Kernel 的安全保证。 |
| `C5_relayout_recheck` | Relayout 使用独立新分配执行逻辑复制，并在进入 Fast 前重新检查全部 Guard 义务。 | `executable_invariant` | pact/relayout.py<br>pact/dispatch.py<br>artifact/e5_correctness.md | test/test_guard_dispatch.py<br>test/test_e5_correctness.py | results/guard_dispatch_report.json<br>results/e3_real_risk_cases.json<br>results/guard_dispatch_report.json | 当前修复域限于明确支持的 CUDA float32 输入；复制开销必须单列。 |
| `C6_real_risk` | 至少两个来源、三个问题模式形成可复现 L1/L2 风险证据，且 L1、L2、L3 分层报告。 | `bounded_empirical` | bench/e3/risk.py<br>bench/e3/specs.py | test/test_e3_risk.py | results/e3_real_risk_cases.json<br>results/e3_real_risk_cases.json | 没有接口承诺的 L1 仅称 Kernel 前置条件敏感性，不称上游漏洞。 |
| `C7_workload_integration` | 两个真实工作负载轨迹执行 14 个不同冻结函数体，已知违约被阻止且系统基线同语义。 | `bounded_empirical` | integration/specs.py<br>integration/worker.py<br>bench/e4/correctness.py<br>bench/e4/baselines.py | test/test_e4_workloads.py<br>test/test_e4_correctness.py<br>test/test_e4_baselines.py | results/e4_workload_manifest.json<br>results/e4_workload_correctness.json<br>results/e4_dispatch_baselines.json<br>results/e4_wsl_diagnostics.json | 性能仅为 WSL2 单机诊断；两套原生 Linux GPU 尚未复核。 |
| `C8_secondary_mechanisms` | 谓词引导、SMT 和成本模型按实证结果定位，零删除或不稳定结果不用于填补主贡献。 | `bounded_empirical_and_zero_result` | bench/e3/mutations.py<br>bench/e3/smt.py<br>bench/e4/diagnostics.py | test/test_e3_mutation.py<br>test/test_e3_smt.py<br>test/test_e4_diagnostics.py | results/e3_mutation_comparison.json<br>results/e3_smt_audit.json<br>results/e4_wsl_diagnostics.json<br>results/e3_mutation_comparison.json<br>results/e3_smt_audit.json<br>results/e4_wsl_diagnostics.json | 谓词引导未优于均匀随机；SMT 只发现 1 个同域等价重复；当前不声称普遍搜索优势、广泛精简或稳定加速。 |
| `C9_fingerprint_fail_closed` | 语料、冻结规则或阶段源码指纹失配时，总核验器拒绝把历史报告视为当前证据。 | `machine_audit` | bench/e1/audit.py<br>bench/e2/freeze.py<br>bench/e3/validate.py<br>bench/e4/validate.py<br>bench/e5/validate.py | test/test_e5_validate.py | results/e4_regression.json<br>results/e5_evidence_manifest.json | 提交后的干净 checkout 仍需重新运行一次总核验，才能满足最终冻结条件。 |

## 机器核验

- [x] all_claims_have_code
- [x] all_claims_have_tests
- [x] all_claims_have_reports
- [x] all_claims_have_raw_data
- [x] all_claims_state_limitations
- [x] all_references_exist
