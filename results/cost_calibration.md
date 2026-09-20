# 第五阶段离线成本标定

生成时间：2026-09-20T22:09:14.126052+08:00
GPU：NVIDIA GeForce RTX 5060 Laptop GPU；PyTorch 2.11.0+cu128；Triton 3.6.0。
预热 3 次、每路径 15 次；JIT 编译排除。

端到端采用同步后的主机时钟；组成段的 GPU 操作采用 CUDA Event，Guard 用主机时钟（直接 Fast 时含实际输出分配与 span 检查）。复制分配单列主机时间。各段之和不等于端到端时间。

| 案例 | 布局 | 可行路径 | 成本选择 | 端到端 median / p95 (µs) | Guard median (µs) | Allocate host median (µs) | Copy Event median (µs) | Kernel Event median (µs) | Fallback Event median (µs) |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| A | contiguous | Fast, PyTorch Fallback | Fast (波动重叠，默认) | Fast: 174.1/234.0; PyTorch Fallback: 60.2/139.0 | 49.8 | — | — | 46.8 | 28.0 |
| A | transpose | Relayout+Fast, PyTorch Fallback | PyTorch Fallback (波动重叠，默认) | Relayout+Fast: 153.5/289.3; PyTorch Fallback: 61.4/122.9 | 32.1 | 8.0 | 11.8 | 36.6 | 27.4 |
| A2 | padded | Fast, PyTorch Fallback | Fast (波动重叠，默认) | Fast: 59.1/110.1; PyTorch Fallback: 50.4/64.4 | 32.4 | — | — | 19.0 | 11.2 |
| B | contiguous | Fast, PyTorch Fallback | Fast (波动重叠，默认) | Fast: 168.5/207.5; PyTorch Fallback: 181.6/224.8 | 62.6 | — | — | 160.7 | 105.3 |
| B | contiguous | Fast, PyTorch Fallback | Fast (波动重叠，默认) | Fast: 53.2/71.9; PyTorch Fallback: 44.9/75.0 | 26.8 | — | — | 21.6 | 11.2 |
| B | offset | Fast, PyTorch Fallback | Fast (波动重叠，默认) | Fast: 76.3/147.9; PyTorch Fallback: 74.5/143.0 | 39.9 | — | — | 42.9 | 30.7 |
| D | x_strided | Relayout+Fast, PyTorch Fallback | PyTorch Fallback | Relayout+Fast: 307.1/355.7; PyTorch Fallback: 135.4/186.6 | 38.4 | 11.5 | 100.4 | 169.9 | 106.0 |
| D | y_strided | Relayout+Fast, PyTorch Fallback | PyTorch Fallback | Relayout+Fast: 87.3/133.2; PyTorch Fallback: 35.5/57.0 | 17.9 | 2.7 | 35.3 | 46.5 | 13.7 |
| holdout_vector | strided | Fast, PyTorch Fallback | Fast (波动重叠，默认) | Fast: 62.4/93.1; PyTorch Fallback: 45.0/61.5 | 26.9 | — | — | 20.0 | 15.5 |
| holdout_matrix | strided | Fast, PyTorch Fallback | Fast (波动重叠，默认) | Fast: 63.2/84.7; PyTorch Fallback: 54.1/56.0 | 32.0 | — | — | 22.4 | 13.8 |

JSON 保留全部原始样本、分位数、指纹和完整表项。在线仅查表；缺失、过期或不适用时使用确定性默认路径。
