# 第五阶段 Guard 与分派隔离报告

生成时间：2026-09-20T22:19:32.591878+08:00
验收：通过。

## 检查

- 通过：八个入口均有当前源码的受支持计划
- 通过：全部隔离样例数值正确
- 通过：两个留出案例直接 Fast
- 通过：Fast、修复及 Fallback 均有实际路径
- 通过：A2 padding 未复制
- 通过：指针提示错位拒绝且索引提示不要求指针对齐

## 逐例路径

| 案例 | 布局 | 策略 | 结果 | 路径 | 直接 Guard | 修复输入 |
| --- | --- | --- | --- | --- | --- | --- |
| A | contiguous | cost | correct | Fast | True | — |
| A | transpose | cost | correct | PyTorch Fallback | False | — |
| A | transpose | prefer_repair | correct | Relayout+Fast | False | X |
| A | single_row | cost | correct | Fast | True | — |
| A | single_col | cost | correct | Fast | True | — |
| A2 | padded | cost | correct | Fast | True | — |
| A2 | transpose | prefer_repair | correct | Relayout+Fast | False | X |
| B | contiguous | cost | correct | Fast | True | — |
| B | contiguous | cost | correct | Fast | True | — |
| B | contiguous | cost | correct | Fast | True | — |
| B | offset | cost | correct | Fast | True | — |
| B | strided | prefer_repair | correct | Relayout+Fast | False | X |
| D | contiguous | cost | correct | Fast | True | — |
| D | x_offset | cost | correct | Fast | True | — |
| D | x_strided | prefer_repair | correct | Relayout+Fast | False | X |
| D | y_strided | prefer_repair | correct | Relayout+Fast | False | Y |
| holdout_vector | strided | cost | correct | Fast | True | — |
| holdout_matrix | strided | cost | correct | Fast | True | — |
| pointer_hint | contiguous | cost | correct | Fast | True | — |
| pointer_hint | offset | cost | correct | PyTorch Fallback | False | — |
| pointer_hint | offset | prefer_repair | correct | Relayout+Fast | False | X |
| index_hint | offset | cost | correct | Fast | True | — |

完整 JSON 含每条候选的用途、来源、证据、求值顺序与短路位置；有限隔离运行不替代支持域内的静态充分规则。
