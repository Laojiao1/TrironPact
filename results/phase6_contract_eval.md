# 第六阶段受限契约判定试运行

生成时间：2026-09-25T18:14:24.201798+08:00

隔离报告通过：True；独立标签 20，明确判定 20，Unknown 0，Unsupported 0。
TP=14 FP=0 TN=6 FN=0；precision=1.0，已判定子集 recall=1.0。
按 Kernel 宏平均 precision=1.0（8 个可定义分母），recall=1.0（8 个可定义分母）。
离线影子候选单列：标签 20，FP=0，FN=0，Unknown=0。
合并仅用于样例总览：标签 40，FP=0，FN=0，Unknown=0。
全清单五层覆盖：{'registered': 15, 'syntax_supported': 14, 'syntax_unknown': 0, 'syntax_unsupported': 1, 'candidate_complete': 14, 'guard_available': 8, 'fast_feasible': 8}。Unsupported 进入清单分母，不进入有标签输入 precision/recall。

| 输入 | 模式 | 独立标签 | 判定 | 路径 | 运行分类 |
| --- | --- | --- | --- | --- | --- |
| A:contiguous:7:11 | online_guard | eligible | True | Fast | correct |
| A:transpose:7:11 | online_guard | ineligible | False | PyTorch Fallback | correct |
| A:single_row:1:11 | online_guard | eligible | True | Fast | correct |
| A:single_col:7:1 | online_guard | eligible | True | Fast | correct |
| A2:padded:7:11 | online_guard | eligible | True | Fast | correct |
| A2:transpose:7:11 | online_guard | ineligible | False | Relayout+Fast | correct |
| B:contiguous:7:127 | online_guard | eligible | True | Fast | correct |
| B:contiguous:7:128 | online_guard | eligible | True | Fast | correct |
| B:contiguous:7:129 | online_guard | eligible | True | Fast | correct |
| B:offset:7:129 | online_guard | eligible | True | Fast | correct |
| B:strided:7:129 | online_guard | ineligible | False | Relayout+Fast | correct |
| D:contiguous:7:129 | online_guard | eligible | True | Fast | correct |
| D:x_offset:7:129 | online_guard | eligible | True | Fast | correct |
| D:x_strided:7:129 | online_guard | ineligible | False | Relayout+Fast | correct |
| D:y_strided:7:129 | online_guard | ineligible | False | Relayout+Fast | correct |
| holdout_vector:strided:7:129 | online_guard | eligible | True | Fast | correct |
| holdout_matrix:strided:7:11 | online_guard | eligible | True | Fast | correct |
| pointer_hint:contiguous:7:128 | online_guard | eligible | True | Fast | correct |
| pointer_hint:offset:7:128 | online_guard | ineligible | False | PyTorch Fallback | correct |
| index_hint:offset:7:128 | online_guard | eligible | True | Fast | correct |
| bench_relu:contiguous | offline_shadow_candidate | eligible | True | Offline Shadow | correct |
| bench_relu:offset | offline_shadow_candidate | eligible | True | Offline Shadow | correct |
| bench_relu:strided | offline_shadow_candidate | ineligible | False | Offline Shadow | numeric_mismatch |
| bench_square:contiguous | offline_shadow_candidate | eligible | True | Offline Shadow | correct |
| bench_square:offset | offline_shadow_candidate | eligible | True | Offline Shadow | correct |
| bench_square:strided | offline_shadow_candidate | ineligible | False | Offline Shadow | numeric_mismatch |
| bench_multiply:contiguous | offline_shadow_candidate | eligible | True | Offline Shadow | correct |
| bench_multiply:offset | offline_shadow_candidate | eligible | True | Offline Shadow | correct |
| bench_multiply:strided | offline_shadow_candidate | ineligible | False | Offline Shadow | numeric_mismatch |
| bench_axpy:contiguous | offline_shadow_candidate | eligible | True | Offline Shadow | correct |
| bench_axpy:offset | offline_shadow_candidate | eligible | True | Offline Shadow | correct |
| bench_axpy:strided | offline_shadow_candidate | ineligible | False | Offline Shadow | numeric_mismatch |
| bench_feature_scale:contiguous | offline_shadow_candidate | eligible | True | Offline Shadow | correct |
| bench_feature_scale:offset | offline_shadow_candidate | eligible | True | Offline Shadow | correct |
| bench_feature_scale:x_col_strided | offline_shadow_candidate | ineligible | False | Offline Shadow | numeric_mismatch |
| bench_feature_scale:feature_strided | offline_shadow_candidate | ineligible | False | Offline Shadow | numeric_mismatch |
| bench_feature_bias:contiguous | offline_shadow_candidate | eligible | True | Offline Shadow | correct |
| bench_feature_bias:offset | offline_shadow_candidate | eligible | True | Offline Shadow | correct |
| bench_feature_bias:x_col_strided | offline_shadow_candidate | ineligible | False | Offline Shadow | numeric_mismatch |
| bench_feature_bias:feature_strided | offline_shadow_candidate | ineligible | False | Offline Shadow | numeric_mismatch |

仅既有确切输入；标签来自索引/调用/提示义务的人工审计，未经第二名审计者盲审。不能外推契约恢复总体 precision/recall。
Precision/recall 仅作用于有独立标签且已明确判定的输入；Unknown/Unsupported 单列，recall 不能解释为全清单召回。
