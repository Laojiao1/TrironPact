# e4a 工作负载正确性与路径

- 状态：`go`；计数：`{'workloads_complete': 2, 'distinct_kernels_executed': 14, 'fast_paths': 14, 'known_false_allows': 0, 'blocked_numeric_mismatches': 2, 'safe_nonstandard_fast_paths': 1, 'safe_outside_frozen_domain_rejections': 1}`。
- 两个工作负载的有效输入全部由冻结 Guard 放行并与独立参考一致。
- 两个已知数值错读布局均被 Guard 拒绝，Fallback 结果正确。
- 项目 A2 padded-row 路径安全直通且不复制；它不计入上游函数体数量。
- Liger softmax row-padding 在固定样例上数值正确，但 e2 exact-stride 声明域拒绝，作为能力缺口单列。

## 核验

- [x] two_workloads_complete
- [x] ten_distinct_kernels_executed
- [x] all_registered_fast_paths_correct
- [x] violations_blocked_and_fallback_correct
- [x] project_safe_nonstandard_fast_without_copy
- [x] upstream_row_padding_gap_disclosed
- [x] zero_known_false_allows

## 证据边界

- **workloads**：两个真实调用轨迹的冻结输入经验核对；14 个不同函数体实际执行。
- **safe_nonstandard**：安全直通来自历史项目 A2 padded-row 路径，不计入 14 个上游函数体。
- **gap**：Liger row-padding 数值正确但 e2 exact-stride Guard 保守拒绝；不授予 Fast，也不隐去复制机会缺口。
