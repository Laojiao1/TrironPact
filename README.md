# TritonPact：Triton Kernel 访存契约分析

TritonPact 研究 PyTorch Tensor 的物理布局与 Triton Kernel 访存之间的契约。项目已完成**第一阶段 PoC**、**第二阶段契约 DSL 与统一 AST/Access IR**、**第三阶段候选契约提取**和**第四阶段边界证据与精化**。在外部给定算子语义、Kernel 参数映射和 PyTorch 参考实现的前提下，系统解析受支持的 Triton Python 源码，核对物理布局候选，再用边界输入、隔离 Oracle 和现有 Guard 验证 Fast / PyTorch Fallback 分派。

当前结论仅适用于已实现的规则 Tile、有限仿射地址模板、显式 mask 和限定输入域。算子原本应实现的语义不能只从 Kernel 当前实现推断；现阶段由 `pact/semantics.py` 明确提供。项目主线见[研究项目说明](../../docs/TritonPact%20研究项目说明.md)，阶段范围与实施记录见[第一阶段 PoC 开发方案](../../docs/开发路线/TritonPact%20第一阶段%20PoC%20开发方案.md)和[第二阶段开发方案](../../docs/开发路线/TritonPact%20第二阶段%20契约DSL与Access%20IR开发方案.md)。

## 当前进度

| 阶段 | 已完成的工作 | 证据 |
| --- | --- | --- |
| 第一阶段：受限 PoC | 四个固定案例的有限候选、边界修订、隔离 Oracle 与 Guard/Fallback | [入库的 PoC 基线](results/poc_results.md)及同名 JSON |
| 第二阶段：契约表示与访存解析 | 有类型的契约 DSL；四例共用的 AST → Access IR 入口；独立语义与参数绑定核对；未知语法和错误绑定拒绝 | [Access IR 验收报告](results/access_ir_report.md)及同名 JSON；[分派回归报告](results/dispatch_regression.md)及同名 JSON |
| 第三阶段：候选提取 | 受限 wrapper 绑定核对；A/A2/B/D 与两个留出 Kernel 的 shape/stride、逐访问点 span 候选；成对 `tl.multiple_of` alignment 义务；离线机器报告与完整隔离回归 | [候选提取报告](results/candidate_extraction_report.md)、[第三阶段分派回归](results/candidate_regression.md)及同名 JSON；[开发记录](../../docs/开发路线/TritonPact%20第三阶段%20候选契约提取开发方案.md) |
| 第四阶段：边界证据与精化 | 受限 JSON 变异配方、独立 GPU worker、原始 Kernel 与参考语义比较、逐候选影子值、保守精化建议和受限 SMT 冗余检查 | [边界证据报告](results/boundary_refinement_report.md)、[第四阶段分派回归](results/refinement_regression.md)及同名 JSON；[开发记录](../../docs/开发路线/TritonPact%20第四阶段%20边界证据与精化开发方案.md) |

第二阶段的 `Supported` 只表示受限访存语法可解释。DSL 已能表达条件和来源，但 Guard 当前仍执行经独立语义核对的 **PoC 兼容谓词**；系统性提取 alignment、shape/stride、offset/span 候选属于第三阶段。

第三阶段新增的候选记录来源、推导规则、支持域和证据状态，只用于离线与影子核对。它们尚未接入 Guard，也不扩大 Fast 范围。shape/stride 与逐访问点 span 规则覆盖已核对调用绑定的一维/二维扁平 Tile、二维逐行 Tile，以及一维显式输入步长和二维显式行/列步长；alignment 只解释独立赋值的正二次幂 `tl.multiple_of` 指针基址或 `pid(0)*Tile` 标量提示。其他形式保留 `Unknown/Unsupported`。

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
| `pact/access_ir.py` | 从受支持的 Triton AST 统一提取、规范化访存地址、mask、Tile、参数位置与源码来源；语法支持不代表 Fast 资格。 |
| `pact/contract_dsl.py` | 用有类型条件树表示输入属性、三类用途、适用域、来源和三值判定。 |
| `pact/candidates.py`、`pact/call_site.py` | 记录离线候选与义务，并核对受限 wrapper 的标量、grid、指针和新建输出绑定。 |
| `pact/shape_stride.py`、`pact/alignment.py`、`pact/span.py` | 从 Access IR 与独立语义提取三类受限候选；逐访存 span 与整视图 span 分开记录。 |
| `pact/candidate_report.py` | 汇总候选、影子求值、拒绝原因及旧 Guard 路径，不参与分派。 |
| `pact/mutation.py`、`pact/mutation_oracle.py`、`scenarios/mutation_worker.py` | 生成有界物理变异，逐例启动独立子进程并返回带实际地址与参考数值的结构化诊断；PoC worker 保持原入口。 |
| `pact/smt_refine.py`、`pact/boundary_report.py` | 在明确整数域内检查条件冗余，汇总执行证据与影子精化建议；不改运行时 Guard。 |
| `pact/analysis.py` | 将统一 Access IR 与独立语义绑定，保留 PoC 的有限候选规则，并编码为 DSL 条件。 |
| `pact/refine.py` | 根据 A 的单谓词边界证据与有限索引推导修订候选。 |
| `pact/runtime.py` | 检查 dtype、shape、storage span 与生成的谓词，执行 Fast 或 PyTorch Fallback。 |
| `pact/oracle.py`、`scenarios/worker.py` | 在独立子进程中执行用例并分类数值错误、异常、超时与未知结果。 |
| `scenarios/kernels.py`、`scenarios/external_add.py`、`scenarios/cases.py` | 保存 Kernel、输入布局构造和 PyTorch 参考计算。 |
| `scenarios/holdouts.py`、`scenarios/alignment_fixtures.py` | 保存两个独立语义留出 Kernel 与指针/索引提示成对样例。 |
| `pact/stage_report.py` | 生成第二阶段的 IR 与拒绝边界报告。 |
| `test/` | DSL、Access IR、绑定拒绝、PoC 行为测试与一键隔离验收。 |
| `results/poc_results.md`、同名 JSON | Git 中保存的第一阶段 PoC 基线报告。 |
| `results/dispatch_regression.md`、同名 JSON | 当前代码上的完整隔离回归证据。 |
| `results/access_ir_report.md`、同名 JSON | 契约表示、访存解析和拒绝边界的验收证据。 |
| `results/candidate_extraction_report.md`、`results/candidate_regression.md` 及同名 JSON | 第三阶段候选、拒绝、影子求值和完整隔离回归证据。 |

每条谓词都记录对应的 load 源文件、行号、语义输入与逻辑读取。`storage_offset()` 以元素计；`data_ptr()` 已是视图首元素的有效地址，storage span 检查不会把 offset 再加到这个指针上。

## 第二阶段的统一表示与解析

`parse_access_ir` 对 A/A2/B/D 使用同一源码解析入口。地址和 mask 以结构化表达式保存，参数按签名位置规范化；变量改名和无歧义的加法换序得到等价 IR。每个访问点保留原始地址、展开地址、规范化地址、mask、源码位置及元素宽度。第三阶段还记录受限 `tl.multiple_of` 的实际表达式、倍数和关联访问点。缺失 mask、间接或非仿射索引、重复赋值、动态控制流、未支持的提示形式和错误指针绑定返回 `Unsupported/Unknown`；旧 PoC Guard 对含提示源码仍拒绝放行。

`extract` 在 IR 可解释后，另行核对独立算子语义和标量绑定，再使用现有 PoC 的有限候选规则。DSL 能表示 stride、size、整除、对齐、storage span 与有限 AND/OR；未推导的条件只作为结构存在，不会授予 Fast 资格。

下节给出查看统一 IR 和生成阶段报告的命令。阶段报告同时生成同名 JSON，并列出四个案例的访存结构、旧 PoC 条件的 DSL 来源和拒绝样例；它与 PoC 数值报告分开保存。

## 环境与运行

已在 WSL2 Ubuntu 20.04 的现有 `triton` 环境运行。项目是 Python 源码项目，无需编译安装；首次执行 Triton Kernel 时由 Triton 即时编译。第四阶段的 `stage-boundary` 和 SMT 测试使用环境中已有的 Z3 Python 包（本次版本 `5.1.0`）；旧阶段 CLI 不依赖它。进入项目根目录并激活环境后运行：

```bash
cd /mnt/f/Project/Paper/Code/TritonPact
python main.py access-ir --kernel A2
python main.py stage-ir --output results/access_ir_report.md
python test/run_tests.py --output results/candidate_regression.md
python main.py stage-candidates --output results/candidate_extraction_report.md
python test/run_tests.py --output results/refinement_regression.md
python main.py stage-boundary --output results/boundary_refinement_report.md
```

- `access-ir` 可用 `--kernel A|A2|B|D` 查看单个案例的统一访存 IR；`ir` 查看与独立语义绑定后的 PoC 分析。
- `stage-ir` 查看第二阶段历史验收；`stage-candidates` 读取第三阶段完整隔离回归，生成候选 Markdown 与同名 JSON。
- `stage-boundary` 使用第三阶段候选生成第四阶段边界证据与 SMT 审计；`--quick` 每例只运行一个代表样例。变异 worker 从标准输入接收受限 JSON，每个子进程只处理一个样例。
- `python test/run_tests.py --output results/candidate_regression.md` 先运行全部阶段测试，再执行关键配置各 10 次的完整隔离验收，报告与前两阶段分开。省略 `--output` 时沿用第二阶段默认路径。
- 单独运行隔离案例可用 `python main.py demo --repeats 10 --output results/manual_demo.md`；`--repeats` 范围为 1～20。

日常快速检查可运行：

```bash
python test/run_tests.py --quick --output results/candidate_regression_quick.md
python main.py stage-boundary --quick --output results/boundary_refinement_quick.md
```

快速模式只运行一轮关键配置；`--output` 将 Markdown 和同名 JSON 另存。也可单独运行 `python -m pytest -q test/test_poc.py`。验收失败时命令返回非零退出码。

## 验收结果与证据边界

第二阶段基线为 **18 项测试、11/11 项 PoC 检查及 4/4 项 IR 检查通过**。第三阶段完整验收为 **50 项测试、11/11 项隔离检查及 6/6 项候选报告检查通过**；机器结果中 `go_core=true`、`refinement_verified=true`、`go_candidates=true`。完整回归保留 28 个逐例运行和 27 个附加重复运行，A 转置原始 Fast、A2 padding 直通、B 非整除直通三组关键配置各累计 10 次。两个留出 Kernel 的非连续输入和成对提示样例另经隔离进程数值核对；有限运行结果不替代静态推导。

第三阶段报告统计 **43 条记录**（含拒绝义务）：23 条为受支持规则下的静态充分条件、1 条动态指针对齐义务为 `Unproven`、19 条为 `Unknown`。该计数表示条件或义务的生成与证据状态；代表性元数据影子求值和旧 Guard 实际路径分别列示，不以候选数量或样例通过授予 Fast 资格。

第四阶段完整边界报告运行 36 个变异：29 个正确、6 个数值错读、1 个因实际有效指针不满足提示而在执行前跳过。已知 A singleton 过强见证及其静态修订理由沿用历史回归，并明确标注来源。SMT 在合成蕴含样例上证明 `ptr % 16 == 0` 蕴含 `ptr % 4 == 0`；真实第三阶段候选没有满足同目的、同域、可完整形式化且冗余的条件对，因此没有删除真实条件。第四阶段仍不接入新 Guard 或扩大 Fast。

机器结果分别记录 `guard_basis`、`fast_eligible` 和 `observation_level`。`Statically-Proven` 只指声明语义、参数映射与受支持模板内的布局判断，且只有 Guard 放行才取得 Fast 资格；`Empirically-Validated` 只说明这一次运行已观察到正确或错误结果。普通异常、超时和进程故障不会直接被判作非法访存。未知 dtype、无法解释的输入或超出支持范围的情况不进入 Fast。

旧 PoC Guard 尚未接入第三阶段新候选，也没有新增对齐提示案例 C 的 Fast 路径。复杂间接索引、动态循环、任意 alias、已有 Generic Kernel、图级 Guard、自动成本分派和库级自动发现与语义恢复仍不在当前实现范围内。A2 证明了一个非连续输入可以避免复制并正确直通；当前报告没有端到端性能测量，也不据此宣称稳定加速。
