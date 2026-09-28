# 第六阶段受限 Guard 与分派性能试运行

生成时间：2026-09-25T11:17:47.984252+08:00
环境：NVIDIA GeForce RTX 5060 Laptop GPU；Torch 2.11.0+cu128；Triton 3.6.0。
测量前设备状态：{'available': True, 'fields': ('temperature_c', 'graphics_mhz', 'memory_mhz', 'power_w', 'limit_w'), 'values': ['84, 262, 9001, 115.97, [N/A]'], 'reason': ''}；测量后：{'available': True, 'fields': ('temperature_c', 'graphics_mhz', 'memory_mhz', 'power_w', 'limit_w'), 'values': ['84, 1687, 9001, 113.52, [N/A]'], 'reason': ''}。
预热 3，交错测量 15 轮；JIT 编译排除。

| 输入 | 直接 Guard | 成本表实际路径 | 各安全路径 p50/p95 (µs) |
| --- | --- | --- | --- |
| A contiguous 9×15 | True | Fast | Fallback: 1910.0/1938.5; DispatchDefault: 1909.5/1964.1; DispatchTable: 1339.4/1918.9; DirectFast: 1131.7/3302.4 |
| A transpose 9×15 | False | PyTorch Fallback | Fallback: 1148.8/1925.9; DispatchDefault: 1076.4/2265.8; DispatchTable: 1908.9/1914.9; RelayoutFast: 1916.4/2011.3 |
| A2 padded 5×13 | True | Fast | Fallback: 1829.7/1917.6; DispatchDefault: 1760.4/1917.1; DispatchTable: 1663.7/1971.8; DirectFast: 1456.7/1921.5 |
| B contiguous 255 | True | Fast | Fallback: 1893.3/1919.8; DispatchDefault: 1904.0/1912.2; DispatchTable: 1412.3/1914.7; DirectFast: 1166.0/1979.8 |
| B strided 257 | False | PyTorch Fallback | Fallback: 1075.7/1914.0; DispatchDefault: 1295.9/1912.1; DispatchTable: 1890.5/1917.7; RelayoutFast: 1888.5/1927.2 |
| D x_strided 257 | False | PyTorch Fallback | Fallback: 1724.0/1921.3; DispatchDefault: 1221.0/1914.2; DispatchTable: 1581.7/1917.7; RelayoutFast: 1914.7/2212.1 |
| holdout_vector strided 257 | True | Fast | Fallback: 1900.4/1926.2; DispatchDefault: 1846.4/1915.0; DispatchTable: 1410.2/1945.3; DirectFast: 1166.8/1919.3 |
| pointer_hint offset 256 | False | PyTorch Fallback | Fallback: 1351.3/2178.0; DispatchDefault: 1172.9/2469.3; DispatchTable: 1910.6/2066.8; RelayoutFast: 1913.4/1927.7 |
| A contiguous 512×512 | True | Fast | Fallback: 1836.9/1925.0; DispatchDefault: 1629.6/1923.5; DispatchTable: 1193.8/1930.7; DirectFast: 1363.5/1967.5 |
| B contiguous 262143 | True | Fast | Fallback: 1081.4/1927.3; DispatchDefault: 1362.2/1930.9; DispatchTable: 1917.2/2102.7; DirectFast: 1914.9/1969.2 |
| D contiguous 262143 | True | Fast | Fallback: 1674.6/1930.5; DispatchDefault: 1186.6/1975.5; DispatchTable: 1864.5/1944.4; DirectFast: 1893.2/1930.1 |
| D x_strided 129 shift=0.25 | False | PyTorch Fallback | Fallback: 1076.6/1917.1; DispatchDefault: 1168.8/1915.2; DispatchTable: 1914.3/1925.0; RelayoutFast: 1916.3/1948.6 |
| D y_strided 129 shift=0.25 | False | PyTorch Fallback | Fallback: 1904.1/1914.3; DispatchDefault: 1906.4/1939.2; DispatchTable: 1586.0/2155.7; RelayoutFast: 1326.7/1977.3 |

成对安全输入 13 个；单次同步 Fallback/DispatchTable 几何平均比 0.930796105803232；不确定性需结合原始样本判断。
按轮次配对 bootstrap 比值 0.948838032542925，条件 95% 区间 (0.7829432524589653, 1.1027617271151935)。
每批 16 次调用、一次同步的摊销吞吐比：0.8929061443318919（非单次调用延迟）。
批量摊销配对 bootstrap 比值 0.9140473267860694，条件 95% 区间 (0.8496328694722712, 1.0001579721846863)。
当前成本表精确匹配：2/13；未命中时执行第五阶段的确定性默认路径。
设备状态只在测量前后读取；图形频率或温度变化可能使稳态假设失效。条件区间不覆盖跨运行和跨硬件波动。

## 摊销批量主机成本 p50/p95 (µs/调用)

| 输入 | 各安全路径 |
| --- | --- |
| A contiguous 9×15 | Fallback: 157.7/240.9; DispatchDefault: 203.8/240.8; DispatchTable: 188.7/292.3; DirectFast: 151.7/240.1 |
| A transpose 9×15 | Fallback: 173.1/241.1; DispatchDefault: 216.6/243.8; DispatchTable: 134.3/256.0; RelayoutFast: 289.4/363.8 |
| A2 padded 5×13 | Fallback: 194.2/241.7; DispatchDefault: 173.0/262.4; DispatchTable: 210.0/241.0; DirectFast: 155.6/240.9 |
| B contiguous 255 | Fallback: 155.9/244.3; DispatchDefault: 203.2/260.8; DispatchTable: 220.0/262.0; DirectFast: 206.3/248.5 |
| B strided 257 | Fallback: 204.1/297.2; DispatchDefault: 188.1/258.6; DispatchTable: 212.0/261.6; RelayoutFast: 280.9/364.9 |
| D x_strided 257 | Fallback: 177.0/242.5; DispatchDefault: 218.4/245.0; DispatchTable: 166.9/260.5; RelayoutFast: 293.4/361.8 |
| holdout_vector strided 257 | Fallback: 155.9/241.8; DispatchDefault: 206.4/287.4; DispatchTable: 169.2/258.1; DirectFast: 191.5/240.1 |
| pointer_hint offset 256 | Fallback: 179.9/238.2; DispatchDefault: 132.9/293.2; DispatchTable: 197.4/255.5; RelayoutFast: 286.7/361.7 |
| A contiguous 512×512 | Fallback: 226.6/244.8; DispatchDefault: 224.8/264.0; DispatchTable: 226.1/261.8; DirectFast: 205.3/270.0 |
| B contiguous 262143 | Fallback: 167.7/242.0; DispatchDefault: 230.8/295.6; DispatchTable: 173.0/274.1; DirectFast: 207.2/245.5 |
| D contiguous 262143 | Fallback: 153.5/243.6; DispatchDefault: 230.8/243.3; DispatchTable: 177.0/269.6; DirectFast: 188.6/242.7 |
| D x_strided 129 shift=0.25 | Fallback: 209.9/294.3; DispatchDefault: 189.4/258.6; DispatchTable: 297.6/369.3; RelayoutFast: 295.4/335.0 |
| D y_strided 129 shift=0.25 | Fallback: 189.4/242.8; DispatchDefault: 160.0/256.7; DispatchTable: 297.2/366.1; RelayoutFast: 273.1/386.7 |

## GuardPlan.evaluate 主机时间

| 输入 | 判定 | p50/p95 (µs) | 已运行阶段 |
| --- | --- | --- | --- |
| B contiguous 255 | True | 34.6/48.3 | device_dtype, rank_shape, stride, span, span |
| B strided 257 | False | 11.9/28.2 | device_dtype, rank_shape, stride |
| pointer_hint offset 256 | False | 16.6/20.5 | device_dtype, rank_shape, stride, pointer |

各安全路径逐轮交错且轮换顺序；每次调用后 torch.cuda.synchronize；主机 perf_counter_ns。输入已构造，输出分配留在调用内。
完整 GuardPlan.evaluate 含实际输出分配；不是纯谓词 CPU 纳秒级测量。
仅历史受限入口的 11 组留出尺寸与 2 组同元数据留出数值；缺失新算子、新来源、真实 SMT 删除消融与独立重复硬件，不代表第六阶段完整基准。
