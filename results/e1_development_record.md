# e1 实施记录

日期：2026-09-30。所有命令均在 `F:\Project\Paper\Code\TritonPact` 对应的 WSL 路径下、既有 `triton` 环境中使用 `python` 运行；未提交、推送、重置或清理 Git 工作树。

| 工作块 | 命令 | 退出码/结果 | 证据位置 |
| --- | --- | --- | --- |
| 开工基线 | Git HEAD/工作区、Python/Torch/Triton/CUDA/GPU 与 `bench.validate` 只读核对 | 0；HEAD=origin/main=`8eb9d97`，开工前干净；第六阶段当前指纹匹配且 `go_stage6=true` | `results/e1_regression.json` 的 `git`、`environment`、`phase6` |
| 初始快速回归 | `python test/run_tests.py --quick --output /tmp/tritonpact_e1_baseline_quick.md` | 0；107 passed；旧隔离 11/11 | 临时开工证据；最终结果由 `results/e1_regression.md/.json` 覆盖复核 |
| 来源固定与清单构建 | `python -m bench.e1.catalog --build` | 0；40 个函数体，来源/许可/源码/参考/封套哈希和绑定通过 | `bench/e1/catalog.json`、`bench/e1/sources.json` |
| 结构化机器检查 | `python -m pytest -q test/test_e1_corpus.py` | 0；8 passed | `test/test_e1_corpus.py`；最终 115 项汇总见 e1 regression |
| 冻结语义 smoke | `python -m bench.e1.audit --freeze --smoke` | 0；28/28 passed；四类为 7/6/7/8；清单 frozen | `results/e1_semantic_smoke.md/.json`、`e1_corpus_inventory`、`e1_semantic_audit`、`e1_capability_gap` |
| 最近邻与外部基线 | `python -m bench.e1.related --run-triton-verify` | 0；固定提交的 512 元素 SAFE 与 500 元素 BUG FOUND 两例复现 | `results/e1_related_work_baselines.md/.json` |
| 原生资源准备 | `python -m bench.e1.resources` | 0；两个不同架构候选和脚本已登记；机器均未取得、smoke 未运行 | `results/e1_native_linux_resources.md/.json` |
| WSL 边界检查 | `python scripts/e1/native_smoke.py` | 预期非零；`Unsupported: native_linux_required` | `results/e1_native_linux_resources.md/.json` 的 WSL 不计入策略 |
| 最终回归 | `python -m bench.e1.validate` | 0；115 passed；旧隔离 11/11；第六阶段指纹当前 | `results/e1_regression.md/.json`、`results/e1_regression_isolation.md/.json` |

阶段核心证据为 `go_e1_core=true`，候选登记和脚本准备满足 e1 资源准备要求，因此 `go_e1=true`。两套原生 Linux GPU 仍是 `not_acquired/not_run`，`native_resource_ready=false`；该实机项按 2026-09-30 用户决定延期，不阻塞 e2，但必须在正式跨硬件实验前重新打开并完成。没有运行 e4 性能实验，也没有修改 Access IR、候选规则、Guard 或在线 Fast。
