# TritonPact 契约 DSL 与 Access IR 验收报告

> 生成时间：2026-09-20 10:11:28 CST
> 表示与解析阶段检查：**通过**

## 检查项

| 检查项 | 结果 |
| --- | --- |
| 四例由同一源码解析入口形成 Access IR | 通过 |
| 四例仍取得 PoC 有限契约 | 通过 |
| 每条 DSL 条件有来源和适用域 | 通过 |
| 拒绝样例没有获得受支持状态 | 通过 |

## 四个 PoC 案例的统一 Access IR

| 案例 | 分析状态 | 访存 | 指针绑定 | 规范化地址 | 规范化 mask | 源码位置 |
| --- | --- | --- | --- | --- | --- | --- |
| A | Supported | load | X → X | `add(lane(param(4)), mul(param(4), pid(0)))` | `lt(add(lane(param(4)), mul(param(4), pid(0))), mul(param(2), param(3)))` | `scenarios/kernels.py:15` |
| A | Supported | store | Y → OUT | `add(lane(param(4)), mul(param(4), pid(0)))` | `lt(add(lane(param(4)), mul(param(4), pid(0))), mul(param(2), param(3)))` | `scenarios/kernels.py:16` |
| A2 | Supported | load | X → X | `add(lane(param(5)), mul(param(4), pid(0)))` | `lt(lane(param(5)), param(3))` | `scenarios/kernels.py:25` |
| A2 | Supported | store | Y → OUT | `add(lane(param(5)), mul(param(3), pid(0)))` | `lt(lane(param(5)), param(3))` | `scenarios/kernels.py:26` |
| B | Supported | load | X → X | `add(lane(param(3)), mul(param(3), pid(0)))` | `lt(add(lane(param(3)), mul(param(3), pid(0))), param(2))` | `scenarios/kernels.py:34` |
| B | Supported | store | Y → OUT | `add(lane(param(3)), mul(param(3), pid(0)))` | `lt(add(lane(param(3)), mul(param(3), pid(0))), param(2))` | `scenarios/kernels.py:35` |
| D | Supported | load | x_ptr → X | `add(lane(param(4)), mul(param(4), pid(0)))` | `lt(add(lane(param(4)), mul(param(4), pid(0))), param(3))` | `scenarios/external_add.py:19` |
| D | Supported | load | y_ptr → Y | `add(lane(param(4)), mul(param(4), pid(0)))` | `lt(add(lane(param(4)), mul(param(4), pid(0))), param(3))` | `scenarios/external_add.py:20` |
| D | Supported | store | output_ptr → OUT | `add(lane(param(4)), mul(param(4), pid(0)))` | `lt(add(lane(param(4)), mul(param(4), pid(0))), param(3))` | `scenarios/external_add.py:22` |

IR 中同时保存原始与展开后的地址、mask、元素宽度及 Tile 参数；完整结构见同名 JSON。`Supported` 只表示访存语法可解释，不授予 Fast 资格。

## 与独立语义绑定的 PoC 条件

| 案例 | 契约状态 | 条件用途 | 逻辑读取与访存来源 |
| --- | --- | --- | --- |
| A | Supported | Semantics | `x[row, col]` ← `scenarios/kernels.py:15` |
| A | Supported | Semantics | `x[row, col]` ← `scenarios/kernels.py:15` |
| A2 | Supported | Semantics | `x[row, col]` ← `scenarios/kernels.py:25` |
| B | Supported | Semantics | `x[i]` ← `scenarios/kernels.py:34` |
| D | Supported | Semantics | `x[i]` ← `scenarios/external_add.py:19` |
| D | Supported | Semantics | `y[i]` ← `scenarios/external_add.py:20` |

这些是旧 PoC 规则在新 IR 上核对后编码成 DSL 的条件；本阶段没有系统提取 alignment、shape/stride 或 offset/span 新候选。

## 拒绝样例

| 输入 | 状态 | 原因 |
| --- | --- | --- |
| 缺少 load mask | Unsupported | tl.load/store 缺少显式 mask |
| 索引量相乘 | Unsupported | 两个索引量相乘不在仿射支持域 |
| 错误指针绑定 | Unknown | 指针绑定缺失、重复或不在函数签名中 |

## 结论边界

- 逻辑语义和 launch 参数映射仍由 PoC 的独立声明提供，不从 Kernel 实现推断算子意图。
- 本报告验证表示、解析与拒绝边界；数值、路径和十次重复的证据仍以 PoC 隔离报告为准。
- 复杂间接索引、动态控制流、`tl.multiple_of`、库级自动发现及系统候选提取均未在本阶段放行。
