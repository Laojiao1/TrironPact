# e5a 开发记录

## 范围

e5a 只实现不依赖外部资源的正确性论证、证据追踪、冻结留出结果只读审计和总核验器。没有修改 e2 冻结规则、在线 Fast 或留出集，也没有执行 e4b 原生 Linux 跨硬件实验。

## 基线核对

- 开始提交：`481c1651fc081f05c6cddc1492e41f05fc6afd9d`（`phase6 e4a`）。
- `eval`、`origin/eval` 和 HEAD 一致；开工时工作区干净。
- WSL2 环境：Python 3.10.20、PyTorch 2.11.0+cu128、Triton 3.6.0、RTX 5060 Laptop GPU。
- e1 语料/来源、e2 冻结规则、e3 源码和 e4a 源码指纹均由最终总核验器重新计算并匹配。

## 可独立核验工作块

1. 支持域与正确性说明：固定受支持 Triton 子语言、Access IR 坐标/单位、外部语义前提、翻译保持、候选充分性、Guard 保守性、Relayout 二次复验和指纹 fail-closed 边界。
2. 留出只读审计：复用 e2 已冻结的一次性评估，未启动 Kernel；7 个留出中 6 Supported/可行、1 Unknown，规则指纹未变化。
3. 主张追踪：9 项主张均绑定代码、测试、报告、原始数据、证据级别和限制。
4. readiness：逐项机器核对 CCF B 十项条件，单独生成 CCF A gap；外部资源与最终冻结条件保持未满足。
5. 总核验器：内容寻址 21 项正式输入资产，核对 Git、报告时间、环境、数据划分、七层分母、误放行、真实风险、系统基线和次要/零结果定位。

## 验证记录

| 命令 | 退出码 | 结果 | 报告 |
| --- | ---: | --- | --- |
| `python -m bench.e5.validate`（首轮） | 2 | 153 passed、2 failed；追踪模块名和既有结果 schema/工作树目录处理不一致，准备 Go 被正确拒绝 | 首轮诊断已由后续正式报告替代，失败原因保留在本记录 |
| `python -m bench.e5.validate`（追踪修正后） | 0 | 155 passed；旧隔离回归 11/11；全部 e5a 准备检查通过 | 中间验收 |
| `python -m bench.e5.validate`（最终） | 0 | 156 passed；旧隔离回归 11/11；未来第二审阅/提交绑定条件改为可验证 schema 后仍通过 | `results/e5_regression.md/.json`、`results/e5_regression_isolation.md/.json` |
| `git diff --check` | 0 | 无空白错误 | 终检命令输出 |

## 状态与边界

- `go_e5_preparation=true`：e5a 范围完成。
- `go_e5_ccfb_readiness=false`、`go_e5=false`：e4b 双原生 Linux GPU、第二审阅人、提交后证据绑定和干净 checkout 最终重跑仍缺失。
- e5a 源码指纹：`c35135b9d053753192483410b7eb3883828630feb29cac8bf36e418d7e9dfe93`。
- e3 结果按正式报告原样定位：谓词引导未优于均匀随机；SMT 发现 1 个同域等价重复，合成删除为 0；不宣称普遍搜索优势或广泛精简。

