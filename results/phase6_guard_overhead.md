# 第六阶段 Guard 开销受限试运行

原始测量时间：2026-09-25T11:17:47.984252+08:00

| 输入 | GuardPlan.evaluate p50/p95 (µs) | 判定 |
| --- | ---: | --- |
| B contiguous 255 | 34.6/48.3 | True |
| B strided 257 | 11.9/28.2 | False |
| pointer_hint offset 256 | 16.6/20.5 | False |

| 输入 | 原始安全 Fast p50 (µs) | 完整分派 p50 (µs) | p50 比 |
| --- | ---: | ---: | ---: |
| A contiguous 9×15 | 1131.7 | 1909.5 | 1.687 |
| A2 padded 5×13 | 1456.7 | 1760.4 | 1.208 |
| B contiguous 255 | 1166.0 | 1904.0 | 1.633 |
| holdout_vector strided 257 | 1166.8 | 1846.4 | 1.582 |
| A contiguous 512×512 | 1363.5 | 1629.6 | 1.195 |
| B contiguous 262143 | 1914.9 | 1362.2 | 0.711 |
| D contiguous 262143 | 1893.2 | 1186.6 | 0.627 |

GuardPlan.evaluate 含输出分配和 span；DirectFast 为仅在静态安全输入运行的原始 wrapper。同步与主机噪声使差值不等于独立谓词成本，不跨输入推断。
