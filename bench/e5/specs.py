"""e5a 论文主张与可追踪证据的有类型清单。"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TraceClaim:
    """一项可审计主张及其实现、测试、报告、原始数据和限制。"""

    id: str
    claim: str
    evidence_level: str
    code: tuple[str, ...]
    tests: tuple[str, ...]
    reports: tuple[str, ...]
    raw_data: tuple[str, ...]
    limitations: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


CLAIMS = (
    TraceClaim(
        "C1_corpus_truth",
        "40 个固定开源函数体具有去重、来源、许可、绑定和语义审计；28 个正向样例具有独立参考。",
        "source_and_bounded_empirical",
        ("bench/e1/catalog.py", "bench/e1/audit.py", "bench/e1/catalog.json"),
        ("test/test_e1_corpus.py",),
        ("results/e1_corpus_inventory.json", "results/e1_semantic_audit.json"),
        ("results/e1_semantic_smoke.json",),
        ("语义审计目前由单人完成；冻结输入 smoke 不构成一般语义证明。",),
    ),
    TraceClaim(
        "C2_access_translation",
        "受支持 AST 在 Access IR 中保留指针基址、元素偏移、mask、元素宽度和访问点来源。",
        "static_supported_subset",
        ("pact/e2_access.py", "bench/e2/coverage.py", "artifact/e5_correctness.md"),
        ("test/test_e2_access_ir.py", "test/test_e2_invariants.py", "test/test_e5_correctness.py"),
        ("results/e2_access_ir_coverage.json",),
        ("bench/e1/catalog.json", "bench/e2/freeze.json"),
        ("仅覆盖文档列出的无歧义 AST 子集；动态地址选择、一般循环和间接索引 fail closed。",),
    ),
    TraceClaim(
        "C3_candidate_sufficiency",
        "候选条件在声明语义、绑定和冻结适用域内足以保证逻辑访问与物理访问兼容。",
        "static_template_plus_external_semantics",
        ("pact/e2_contracts.py", "bench/e2/contracts.py", "bench/e2/plans.py", "artifact/e5_correctness.md"),
        ("test/test_e2_contracts.py", "test/test_e2_invariants.py", "test/test_e5_correctness.py"),
        ("results/e2_candidate_extraction.json",),
        ("results/e1_semantic_audit.json", "bench/e2/freeze.json"),
        ("充分性依赖外部语义和完整 wrapper 绑定；reduction 只证明访问域，不证明一般数值规约。",),
    ),
    TraceClaim(
        "C4_guard_fail_closed",
        "Guard 仅在全部义务为 True 时放行；False 或 Unknown 均不进入对应 Fast。",
        "executable_invariant",
        ("pact/e2_guard.py", "pact/guard_plan.py", "pact/dispatch.py", "artifact/e5_correctness.md"),
        ("test/test_e2_guard.py", "test/test_guard_dispatch.py", "test/test_e5_correctness.py"),
        ("results/e2_guard_coverage.json", "results/e4_workload_correctness.json"),
        ("results/e2_guard_smoke.json",),
        ("零已知误放行只适用于冻结样例与声明域，不是任意 Triton Kernel 的安全保证。",),
    ),
    TraceClaim(
        "C5_relayout_recheck",
        "Relayout 使用独立新分配执行逻辑复制，并在进入 Fast 前重新检查全部 Guard 义务。",
        "executable_invariant",
        ("pact/relayout.py", "pact/dispatch.py", "artifact/e5_correctness.md"),
        ("test/test_guard_dispatch.py", "test/test_e5_correctness.py"),
        ("results/guard_dispatch_report.json", "results/e3_real_risk_cases.json"),
        ("results/guard_dispatch_report.json",),
        ("当前修复域限于明确支持的 CUDA float32 输入；复制开销必须单列。",),
    ),
    TraceClaim(
        "C6_real_risk",
        "至少两个来源、三个问题模式形成可复现 L1/L2 风险证据，且 L1、L2、L3 分层报告。",
        "bounded_empirical",
        ("bench/e3/risk.py", "bench/e3/specs.py"),
        ("test/test_e3_risk.py",),
        ("results/e3_real_risk_cases.json",),
        ("results/e3_real_risk_cases.json",),
        ("没有接口承诺的 L1 仅称 Kernel 前置条件敏感性，不称上游漏洞。",),
    ),
    TraceClaim(
        "C7_workload_integration",
        "两个真实工作负载轨迹执行 14 个不同冻结函数体，已知违约被阻止且系统基线同语义。",
        "bounded_empirical",
        ("integration/specs.py", "integration/worker.py", "bench/e4/correctness.py", "bench/e4/baselines.py"),
        ("test/test_e4_workloads.py", "test/test_e4_correctness.py", "test/test_e4_baselines.py"),
        ("results/e4_workload_manifest.json", "results/e4_workload_correctness.json", "results/e4_dispatch_baselines.json"),
        ("results/e4_wsl_diagnostics.json",),
        ("性能仅为 WSL2 单机诊断；两套原生 Linux GPU 尚未复核。",),
    ),
    TraceClaim(
        "C8_secondary_mechanisms",
        "谓词引导、SMT 和成本模型按实证结果定位，零删除或不稳定结果不用于填补主贡献。",
        "bounded_empirical_and_zero_result",
        ("bench/e3/mutations.py", "bench/e3/smt.py", "bench/e4/diagnostics.py"),
        ("test/test_e3_mutation.py", "test/test_e3_smt.py", "test/test_e4_diagnostics.py"),
        ("results/e3_mutation_comparison.json", "results/e3_smt_audit.json", "results/e4_wsl_diagnostics.json"),
        ("results/e3_mutation_comparison.json", "results/e3_smt_audit.json", "results/e4_wsl_diagnostics.json"),
        ("谓词引导未优于均匀随机；SMT 只发现 1 个同域等价重复；当前不声称普遍搜索优势、广泛精简或稳定加速。",),
    ),
    TraceClaim(
        "C9_fingerprint_fail_closed",
        "语料、冻结规则或阶段源码指纹失配时，总核验器拒绝把历史报告视为当前证据。",
        "machine_audit",
        ("bench/e1/audit.py", "bench/e2/freeze.py", "bench/e3/validate.py", "bench/e4/validate.py", "bench/e5/validate.py"),
        ("test/test_e5_validate.py",),
        ("results/e4_regression.json",),
        ("results/e5_evidence_manifest.json",),
        ("提交后的干净 checkout 仍需重新运行一次总核验，才能满足最终冻结条件。",),
    ),
)

