# e1 阶段状态与验收记录

生成时间：2026-09-30T13:30:14.743914+08:00
状态：`go_native_resources_deferred`；核心语料 Go=True；e1 Go=True；原生实机就绪=False。

## 核心核验

| 核验项 | 结果 |
| --- | --- |
| phase6_current_and_go | 通过 |
| regression_and_legacy_isolation | 通过 |
| corpus_frozen | 通过 |
| source_size_threshold | 通过 |
| positive_reference_threshold | 通过 |
| all_positive_smoke_passed | 通过 |
| source_and_binding_audit | 通过 |
| related_work_reviewed | 通过 |
| no_online_capability_change | 通过 |
| semantic_report_current | 通过 |

## 原生资源核验

| 核验项 | 结果 |
| --- | --- |
| two_candidates_registered | 通过 |
| scripts_prepared | 通过 |
| two_native_resources_acquired | 未满足 |
| two_native_smokes_passed | 未满足 |

## 命令与退出码

- `python test/run_tests.py --quick --output /mnt/f/Project/Paper/Code/TritonPact/results/e1_regression_isolation.md`：退出码 0；115 passed in 48.09s；旧隔离 11/11。
- `python -m bench.e1.audit --freeze --smoke`：报告中的 28/28 正向 smoke 与当前指纹一致。
- `python -m bench.e1.related --run-triton-verify`：公开 safe/bug 示例通过，详见相关工作 JSON。
- `python -m bench.e1.resources`：退出码 0；候选登记完成，实际机器状态仍为 not_acquired/not_run。

- 原生实机项由用户决定延期，不阻塞 e2；项目开发完成后、正式跨硬件实验前必须重新打开。

e1 Go 不授予任何新增 Guard/Fast；两套原生 Linux 机器未取得，native_resource_ready 保持 false，正式跨硬件实验前必须重新验收。
