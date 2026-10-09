# TritonPact 第二审阅人独立复核指南

## 1. 冻结对象

- 被审提交：`bd5fa3e3376e2dacffc2671ee52f3cbea1589a10`（`phase6 e5a`）。
- 正式 Kernel 分母：28；本次预先固定 7 个，不得按结果更换样例。
- 风险案例：4 个全部审阅，不得删除失败、Unknown 或分歧案例。
- 清单：`artifact/e5_review/review_manifest.json`；空白记录：`artifact/e5_review/review_template.json`。

审阅人可以使用匿名 reviewer ID，但必须确认未参与 TritonPact 的规则设计、语义标注和预期答案编写。审阅可以在同一台 WSL/GPU 上进行；这属于独立判断复核，不替代 e4b 跨硬件实验。

## 2. 第一轮：盲审并冻结初始判断

先检出冻结提交。只阅读清单给出的上游源码、上游参考文件、`bench/e1/catalog.json` 中对应条目的输入、容差和 wrapper 绑定，以及必要的执行代码。初始判断冻结前不要阅读以下项目结论：

- `results/e1_semantic_audit.*`；
- `results/e2_access_ir_coverage.*`、`e2_candidate_extraction.*`、`e2_guard_coverage.*`；
- `results/e3_real_risk_cases.*`；
- `results/e5_ccfb_readiness.*`。

逐 Kernel 填写：

- `semantic_correspondence`、`wrapper_binding`、`reference_truth`：`confirmed/rejected/unknown`；
- `access_label`：`Supported/Unknown/Unsupported`；
- 至少两条 `evidence`，必须能定位文件、函数、行或独立运行输出；
- 带时区的 `completed_at`，并把 `frozen_before_reconciliation` 设为 `true`。

逐风险案例可执行清单中的 `worker_command`，填写：

- `observed_behavior`：`numeric_mismatch/correct/exception/timeout/unknown`；
- `reference_truth`：`confirmed/rejected/unknown`；
- `guard_behavior`：`blocked/allowed/unknown`；
- `risk_levels`：`L1/L2/L3` 的非空子集；
- `upstream_vulnerability_claim`：`confirmed/rejected/not_claimed/unknown`；
- 至少两条独立证据和初始判断冻结时间。

没有上游接口承诺时，L1 只能称为 Kernel 前置条件敏感性。L2 必须定位真实 wrapper 防御或复制源码；L3 必须核对公开 Issue/PR、状态和提交。

## 3. 第二轮：揭示项目标签与解决分歧

只有第一轮全部冻结后，才阅读项目现有报告并填写 `reconciliation`：

- `project_label_revealed_after_initial=true`；
- 无分歧时设置 `agrees_with_project=true`、`resolution_status=not_needed`；
- 有分歧时设置 `agrees_with_project=false`、`resolution_status=resolved`，并提供 `resolution_evidence`；
- `final_status` 使用 `confirmed/rejected/unknown`，不得为通过验收而强制改成一致。

每个 record-level 分歧必须在顶层 `disagreements` 中恰好有一条记录，保存初始值、项目值、裁决、证据和 `resolved=true`。未解决分歧应保留，但 `disagreements_resolved` 必须为 `false`，此时 readiness 正确保持未满足。

## 4. 交付与机器核验

将空白模板复制为 `results/e5_second_review.json` 后填写，不修改冻结清单。完整记录应满足：

- `status=complete`；
- `reviewed_kernel_count=7`；
- `reviewed_risk_case_ids` 精确包含清单中的四项；
- 7 个不同 Kernel 和 4 个不同风险案例逐项完整；
- `disagreements_resolved=true` 仅在全部分歧确已解决时填写。

运行：

```bash
python -m bench.e5.review validate --input results/e5_second_review.json
```

退出码 0 只表示记录结构、冻结哈希、覆盖和分歧闭环完整；审阅内容本身仍由审阅人负责。建议另外生成 `results/e5_second_review.md`，保留适合人工阅读的结论和签字说明。

