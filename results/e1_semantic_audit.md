# e1 来源、语义与绑定审计

{
  "sources_and_bindings": true,
  "size_and_sources": true,
  "positive_reference_threshold": true,
  "all_declared_positive_smoke": true,
  "frozen": true
}

单人源码/语义核对 + 两种冻结输入隔离核对；第二审阅者未完成

Liger 保存黄金函数；Triton/PyTorch/Unsloth 保存正式测试的公式和构造；本地参考改写逐例记录；sub/masked_add 为独立 Eager 规格，不冒称上游黄金测试

Liger RMSNorm 明确继承 Unsloth；两者是不同函数体，但不视为独立算法来源。PyTorch add 的同构副本未重复登记。

完整逐参数、输入、输出、grid、参考、适配和来源哈希见 bench/e1/catalog.json；逐输入结果与 stderr 见 e1_semantic_smoke.json。
