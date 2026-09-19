# TritonPact 受限研究 PoC

TritonPact 研究 PyTorch Tensor 的物理布局与 Triton Kernel 访存之间的契约。本目录是两周 PoC：在**外部给定算子语义、Kernel 参数映射和 PyTorch 参考实现**的前提下，从受支持的 Triton Python AST 提取访存信息与候选布局条件，再用边界输入、隔离 Oracle 和 Guard 验证 Fast / PyTorch Fallback 分派。

PoC 的结论仅适用于已实现的规则 Tile、有限仿射地址模板、显式 mask 和限定输入域。算子原本应实现的语义不能只从 Kernel 当前实现推断；现阶段由 `pact/semantics.py` 明确提供。项目主线与后续阶段见[研究项目说明](../../docs/TritonPact%20研究项目说明.md)，PoC 验收条件见[两周 PoC 开发方案](../../docs/开发路线/TritonPact%20两周%20PoC%20开发方案.md)。

## 案例与预期行为

| 案例 | Kernel 访存 | 边界输入与分派结果 |
| --- | --- | --- |
| A：二维线性读取 | 把二维输入视作行优先连续地址 | 连续输入进入 Fast；转置和行间 padding 由 Guard 拦截并走 Fallback。转置输入若强行执行原始 Fast，会出现可复现的数值错读。 |
| A2：显式行步长读取 | 行地址使用传入的 `stride(0)`，列地址按单位步长递增 | 行间 padding、列内连续的非连续视图直接进入 Fast，输出正确，输入没有物化；转置视图回退。 |
| B：一维尾部 mask | `tl.load/store` 用 `idx < N` 排除 Tile 尾部 | `N=127、128、129` 均正确直通 Fast，不要求 `N % B == 0`；非单位步长输入回退。 |
| D：双输入向量加法 | 取自 [Triton 官方教程](https://github.com/triton-lang/triton/blob/main/python/tutorials/01-vector-add.py)的两个 load 与一个 store | 分别提取 X、Y 的 stride 条件；任一输入步长为 2 时，原始 Fast 会错读，Guard 拦截并回退；带 offset 且单位步长的输入可直通。 |

A 的初始候选为 `stride(0)==size(1)` 与 `stride(1)==1`。单行、单列输入分别只违反其中一条，隔离执行却正确。边界证据触发候选修订，并由有限索引范围推导支持：`size(0)==1` 时行步长无作用，`size(1)==1` 时列步长无作用。B 的尾部 mask 在初始静态分析中已被识别，**没有**发生动态修订。

## 代码结构

| 位置 | 职责 |
| --- | --- |
| `pact/semantics.py` | 保存 A/A2/B/D 的逻辑索引、预期读取与写入、参数映射、支持域和参考入口；不预填目标 stride 谓词。 |
| `pact/analysis.py` | 从受支持的 Triton AST 提取 Access IR、地址式、mask、源码位置及候选谓词。 |
| `pact/refine.py` | 根据 A 的单谓词边界证据与有限索引推导修订候选。 |
| `pact/runtime.py` | 检查 dtype、shape、storage span 与生成的谓词，执行 Fast 或 PyTorch Fallback。 |
| `pact/oracle.py`、`scenarios/worker.py` | 在独立子进程中执行用例并分类数值错误、异常、超时与未知结果。 |
| `scenarios/kernels.py`、`scenarios/external_add.py`、`scenarios/cases.py` | 保存 Kernel、输入布局构造和 PyTorch 参考计算。 |
| `test/test_poc.py`、`test/run_tests.py` | 测试与一键隔离验收。 |
| `results/poc_results.md`、同名 JSON | 人可读报告与逐例机器数据。 |

每条谓词都记录对应的 load 源文件、行号、语义输入与逻辑读取。`storage_offset()` 以元素计；`data_ptr()` 已是视图首元素的有效地址，storage span 检查不会把 offset 再加到这个指针上。

## 环境与运行

已在 WSL2 Ubuntu 20.04 的现有 `triton` 环境运行。PoC 是 Python 源码项目，无需编译安装；首次执行 Triton Kernel 时由 Triton 即时编译。进入项目根目录并激活环境后运行：

```bash
cd /mnt/f/Project/Paper/Code/TritonPact
python main.py ir --kernel A2
python main.py demo --repeats 10 --output results/poc_results.md
python test/run_tests.py
```

- `ir` 可用 `--kernel A|A2|B|D` 查看单个案例的结构化分析；
- `demo` 运行隔离案例并写入 Markdown 与同名 JSON。
- `--repeats` 范围为 1～20。
- `python test/run_tests.py` 先运行 PoC 测试，再执行关键配置各 10 次的完整隔离验收。

日常快速检查可运行：

```bash
python test/run_tests.py --quick
```

快速模式只运行一轮关键配置，写入 `results/poc_results_quick.md` 与同名 JSON。也可单独运行 `python -m pytest -q test/test_poc.py`。验收失败时命令返回非零退出码。

## 验收结果与证据边界

当前完整验收为 **8 项测试通过、11/11 项报告检查通过**；机器结果中 `go_core=true`、`refinement_verified=true`。报告记录 28 个逐例运行和 27 个附加重复运行，A 转置原始 Fast、A2 padding 直通、B 非整除直通三组关键配置各累计 10 次。数值错误对照只在独立子进程中执行；报告中的 A、D 错读是当前受限案例的观察结果。

机器结果分别记录 `guard_basis`、`fast_eligible` 和 `observation_level`。`Statically-Proven` 只指声明语义、参数映射与受支持模板内的布局判断，且只有 Guard 放行才取得 Fast 资格；`Empirically-Validated` 只说明这一次运行已观察到正确或错误结果。普通异常、超时和进程故障不会直接被判作非法访存。未知 dtype、无法解释的输入或超出支持范围的情况不进入 Fast。

PoC 尚未覆盖对齐提示案例 C、复杂间接索引、动态循环、任意 alias、已有 Generic Kernel、图级 Guard、自动成本分派或库级自动语义恢复。A2 证明了一个非连续输入可以避免复制并正确直通；当前报告没有端到端性能测量，也不据此宣称稳定加速。
