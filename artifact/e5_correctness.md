# TritonPact 支持域与正确性论证

本文固定增强阶段五 a 的支持边界和证明义务。它解释实现为何在受支持域内保守工作，不把有限 GPU 运行、AST 可解析或元数据快筛提升为一般安全证明。

## 1. 受支持子语言与外部前提

分析入口只接受能够唯一绑定到 wrapper 实参的 Triton 函数。受支持表达式包括整数常量、已绑定 `tl.constexpr`、`program_id(0/1)`、`tl.arange`、无歧义整数 cast、有限加减乘、受限 reshape/ravel、行列边界比较及其 AND 组合。每个 `tl.load`/`tl.store` 必须恢复唯一指针基址、元素偏移、mask、元素宽度、输入/输出角色和源码位置。二维 grid、多个访问点、显式 stride、广播和已登记的一类行级 reduction 可以进入 Access IR。

动态地址 `tl.where`、一般间接索引、无法静态展开的循环、复杂布尔 mask、未知 alias、缺失 dtype/元素宽度、错误 grid、指针或标量绑定，以及未建模的数值 reduction 返回 `Unknown/Unsupported`。`Supported` 只说明访存语法在本子集内可解释；它本身不授予 Guard 或 Fast 资格。

外部语义规格必须给出逻辑输入输出、shape/dtype 域、wrapper 全参数绑定、grid、独立参考和容差。输入输出默认使用不同 storage；没有别名证明时不推断别名安全。`data_ptr()` 是视图首元素的有效字节地址；`storage_offset()` 以元素计，只用于元数据和 storage span 推导，不能再次叠加到 `data_ptr()`。

## 2. Access IR 语义

一个访问点表示为 `(kind, base, offset, mask, element_bytes, origin)`。`offset` 是相对 `base` 的元素坐标，乘 `element_bytes` 后才得到字节位移；mask 为假的 lane 不产生物理访存义务。grid 轴和逻辑域来自已核对 wrapper，不能仅由 Kernel 源码猜测。

翻译按 AST 结构递归规范化。变量替换、无歧义加法换序和受限 reshape 不改变元素坐标；指针加法保留唯一 `base` 并把其余项归入 `offset`；mask 只在支持的布尔构造中归一化。任何节点不能满足这些前提时立即停止并保留首个拒绝原因，因此不会用部分 IR 授予 Fast。

**翻译保持草图。** 对受支持表达式作结构归纳：常量、已绑定标量、program id 和 lane 的解释与 Triton 坐标一致；加减乘由归纳假设保持整数值；受限 cast 不改变整数地址语义；reshape/ravel 只改变 block tensor 形状而不改变已恢复的扁平元素坐标。访问点的 `base + offset` 与原 AST 指针表达式逐 lane 相同，规范化 mask 与原 mask 同值。因此，在已绑定 grid/constexpr 域内，Access IR 保持每个活动 lane 的地址和访存使能。未覆盖构造不适用该结论。

## 3. 候选充分性

候选条件按访问点生成，并携带来源、用途、绑定和适用域。shape/stride 条件保证逻辑坐标映射到 Kernel 假设的元素偏移；alignment 条件使用本次有效 `data_ptr()` 和访问表达式的字节倍数；span 条件证明所有 mask 为真的访问字节区间落在实际 storage 内。输出义务与输入义务分别检查。

**充分性草图。** 在外部语义、wrapper 绑定和 Access IR 翻译保持均成立时，若每个逻辑读写坐标满足对应 shape/stride 映射、每个所需地址满足 alignment，且所有活动 lane 的字节区间位于 storage span 内，则 Kernel 的物理访问与声明逻辑访问兼容。该结论是已登记模板和适用域内的充分条件，不主张必要性；保守拒绝可以存在。reduction 样例的独立参考验证数值，但当前静态论证仅覆盖其访问域。

## 4. Guard 与分派

Guard 先检查 Tensor 集合、CUDA 设备、dtype、shape/stride、元素宽度和视图偏移，再检查 alias、逐访问点候选、输出义务及指纹。三值结果中只有 `True` 允许相应 Fast；`False` 进入安全 Fallback 或明确支持的 Relayout，`Unknown` 也不进入 Fast。缺失绑定、证据状态不足或条件无法求值必须保持 `Unknown`。

**保守性草图。** GuardPlan 的编译要求每个候选具有可执行条件、静态证据、来源、绑定和适用域。运行时按合取顺序求值，任一条件不是 `True` 即停止。因此 Fast 意味着该计划内全部前提本次均为真；False/Unknown 不可能经该入口到达对应 Fast。有限样例上的零误放行只验证实现实例，不把草图扩张为支持域外证明。

## 5. Relayout 与二次复验

Relayout 只处理明确标记为可修复的输入义务。实现用 `torch.empty(shape, device, dtype)` 创建独立 storage，再用 `copy_` 按逻辑坐标复制；这避免 `contiguous()` 对某些 offset 视图原样返回。每复制一个输入后重新运行完整 Guard，只有二次结果为 `True` 才允许 Fast。未知 dtype、设备、候选用途或无法继续修复时返回 False/Unknown。

## 6. 指纹与过期证据

e1 绑定语料、来源账本和审计实现；e2 绑定冻结规则、清单和测试；e3/e4 分别绑定本阶段实现、测试和上游输入证据。e5 总核验器重新计算这些指纹，并保存正式 JSON/原始数据的 SHA-256 清单。任一文件缺失、JSON 无法解析、指纹失配、划分重叠或报告时间无效都会使准备检查失败，不会沿用旧 Go。

开发工作树允许仅出现 e5a 与 README/路线记录，但这不等于最终干净 checkout。一旦 e5a 被提交，必须在该提交上重新运行总核验；最终 CCF B readiness 还必须取得两套原生 Linux GPU 和第二审阅人证据。

## 7. 声明边界

- 静态结论：仅限上述子语言、外部语义、完整绑定和候选适用域。
- 运行时结论：Guard/Relayout 的控制流保证只在本次全部义务为真时进入对应 Fast。
- 经验结论：冻结输入、风险变异和工作负载执行只说明已观测案例；不能推出生态级无误放行。
- 性能结论：当前 WSL2 数据是诊断结果，未形成稳定加速或跨硬件结论。
- 未完成证据：第二审阅人和两套原生 Linux GPU 复核仍缺失，因此 e5a 不授予最终 readiness Go。

