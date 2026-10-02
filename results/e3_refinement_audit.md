# e3 精化与 Fast 撤销审计

- 生成时间：2026-10-02T13:18:23.108104+08:00。
- 状态：`go`。
- 分类：`{'expected_rejection': ['triton_add_x_stride', 'liger_softmax_x_stride', 'unsloth_layernorm_x_stride', 'unsloth_layernorm_affine_stride'], 'understrong': [], 'overstrong': [], 'mixed_unattributed': ['grid_mixed', 'triton_two_inputs', 'unsloth_mixed'], 'unreachable': ['span_one_short'], 'unknown': []}`。
- 实际候选修订：0；Fast 撤销：0。
- 边界：e2 Guard 只承诺冻结调用域；域外正确样例不标为过强。若未来出现 Guard=True 的确定错读，必须先撤销 Fast 再继续。
