# e5 第二审阅准备记录

## 冻结范围

- 被审提交：`bd5fa3e3376e2dacffc2671ee52f3cbea1589a10`（`phase6 e5a`）。
- 预先固定 7/28 个正式 Kernel，覆盖 Triton、PyTorch、Liger、FlagGems、Unsloth 五个来源，覆盖四类结构和 development/holdout 两种划分。
- `e3_real_risk_cases.json` 中 4 个 complete 风险案例全部进入审阅分母。
- 清单中的源码、参考和风险报告 SHA-256 从冻结 Git object 读取，不使用未提交工作树替代。

## 产物

- `artifact/e5_review/review_manifest.json`：不含项目结论的盲审输入清单。
- `artifact/e5_review/review_template.json`：7 个 Kernel、4 个风险案例和分歧记录的空白模板。
- `artifact/e5_review/review_guide.md`：两轮审阅、允许值、证据和交付说明。
- `bench/e5/review.py`：冻结包生成与严格完成记录验证器。
- `results/e5_second_review_package.md/.json`：审阅包准备状态。

## 验证

| 命令 | 退出码 | 结果 |
| --- | ---: | --- |
| `python -m bench.e5.review package --freeze-commit bd5fa3e...` | 0 | 7 个 Kernel、4 个风险案例；冻结 blob 哈希匹配 |
| `python -m pytest -q test/test_e5_review.py test/test_e5_readiness.py test/test_e5_traceability.py` | 0 | 10 passed |
| `python -m bench.e5.validate` | 0 | 161 passed；旧隔离 11/11；`go_e5_preparation=true` |
| `git diff --check` | 0 | 无空白错误；仅有 Git 的 LF/CRLF 提示 |

## 状态边界

- 审阅包：`ready_for_independent_reviewer`。
- 第二审阅：尚未开始，`review_complete=false`；用户于 2026-10-09 决定延期到最终文字/证据审核阶段，延期不等于豁免或通过。
- readiness 条件 7：仍为 false；空白模板、重复 Kernel、清单篡改、风险遗漏和未解决分歧均会 fail closed。
- 当前 e5 源码指纹：`820b6192db181ad023d0c19462742e6d9ed8b23a0afb1c6e465230c5f2bec5be`。

