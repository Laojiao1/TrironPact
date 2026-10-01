# e2 实施记录

日期：2026-10-01。起点为用户提交的 `4faf8fb phase6 e1`；开发使用既有 WSL2 `triton` 环境与 RTX 5060 Laptop GPU。未提交、推送、重置或清理工作树。

| 工作块 | 命令 | 退出码/结果 | 证据位置 |
| --- | --- | --- | --- |
| 开工基线 | `python test/run_tests.py --quick --output /tmp/tritonpact_e2_baseline.md` | 0；115 passed；旧隔离 11/11 | 临时开工报告；最终回归重新核验 |
| 绑定后 Access IR | e2/旧 Access IR 相关 pytest；`python -m bench.e2.coverage` | 0；开发集 21/21 Supported | `results/e2_access_ir_coverage.md/.json` |
| 逐访问点候选 | e2 契约及旧 candidate/span pytest；`python -m bench.e2.contracts` | 0；开发集 21/21 候选完整 | `results/e2_candidate_extraction.md/.json` |
| Guard 编译与隔离 | e2 Guard 及旧 Guard/dispatch pytest；`python -m bench.e2.guards --smoke` | 0；开发集 Guard/Fast 可行 21/21 | `results/e2_guard_coverage.md/.json`、`results/e2_guard_smoke.json` |
| 开发规则冻结回归 | `python test/run_tests.py --quick --output /tmp/tritonpact_e2_development_freeze.md` | 0；126 passed；旧隔离 11/11 | 临时冻结前报告；`bench/e2/freeze.json` 保存规则指纹 |
| 唯一一次留出评估 | `python -m bench.e2.coverage --evaluate-holdout`；`python -m bench.e2.contracts --evaluate-holdout`；`python -m bench.e2.guards --evaluate-holdout --smoke` | 0；全正向集 27/28 Supported、候选完整、Guard 可用与 Fast 可行；`flag_slice` 为 Unknown | 三份 e2 正式覆盖报告与 `bench/e2/freeze.json` |
| 验收器口径修复 | 前两次 `python -m bench.e2.validate` | 测试均通过；分别发现 porcelain 前导空格被 `strip()` 删除、WSL 仅行尾差异；只修验收器，冻结规则指纹未变化 | Git 状态检查和最终 `results/e2_regression.json` |
| 最终验收 | `python -m bench.e2.validate` | 0；127 passed；旧隔离 11/11；`go_e2=true` | `results/e2_regression.md/.json`、`results/e2_regression_isolation.md/.json` |

e2 已按 CCF B 关门规则冻结。七层分母为登记 40、正向 28、Supported 27、候选完整 27、Guard 可用 27、Fast 可行 27、集成 0，挑战集 12 保留。Fast 可行只表示冻结输入上的离线 Guard+原始 Kernel+独立参考核对，不注册在线 Fast。reduction 只静态证明访问域，有限数值 smoke 不提升为一般数值证明。TTIR、复杂静态循环、组合 reduction 数值证明和图级 Guard 保留在 CCF A backlog。
