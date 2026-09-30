"""人工选型的 e1 冻结配方；只声明语义，不填写待恢复的布局谓词。"""

from __future__ import annotations

from dataclasses import dataclass

from bench.e1.specs import Binding, SmokeSpec, TensorSpec


@dataclass(frozen=True)
class Selection:
    id: str
    source: str
    file: str
    function: str
    family: str
    test_file: str
    reference_anchor: str
    smoke: SmokeSpec | None
    helper: str = ""
    audit_note: str = ""


CORE = "python/test/unit/language/test_core.py"
STANDARD = "python/test/unit/language/test_standard.py"
UTILS = "torch/testing/_internal/triton_utils.py"
PTTEST = "test/inductor/test_triton_kernels.py"


def tensor(name: str, shape: tuple[int, ...], *, output: bool = False, dtype: str = "float32", fill: str = "random") -> TensorSpec:
    return TensorSpec(name, shape, dtype, "empty" if output else fill)


def spec(tensors: tuple[TensorSpec, ...], inputs: str, outputs: str, bindings: str, grid: tuple[int, ...], reference: str,
         rtol: float = 1e-5, atol: float = 1e-6) -> SmokeSpec:
    return SmokeSpec(tensors, tuple(inputs.split()) if inputs else (), tuple(outputs.split()),
                     tuple(Binding(*p.split("=", 1)) for p in bindings.split(";")), grid, reference, rtol, atol)


def selected() -> tuple[Selection, ...]:
    """28 个正向候选与 12 个挑战；真正分类依据值语义及访存结构。"""
    rows: list[Selection] = []

    def add(id: str, source: str, file: str, function: str, family: str, test: str, anchor: str,
            smoke: SmokeSpec | None, helper: str = "", note: str = "") -> None:
        rows.append(Selection(id, source, file, function, family, test, anchor, smoke, helper, note))

    ew, layout, feature, reduction = "elementwise_mapping", "layout_2d", "feature_broadcast", "row_reduction"
    vector = (tensor("x", (129,)), tensor("y", (129,)), tensor("out", (129,), output=True))
    add("triton_add", "triton", "python/tutorials/01-vector-add.py", "add_kernel", ew,
        "python/tutorials/01-vector-add.py", "output_torch = x + y", spec(vector, "x y", "out", "x_ptr=x;y_ptr=y;output_ptr=out;n_elements=x.numel();BLOCK_SIZE=128", (2,), "add"))
    add("triton_where_zero", "triton", CORE, "test_where_broadcast.where_scalar_condition", ew, CORE, "z = np.where(0, x, 0)",
        spec((tensor("x", (32, 32)), tensor("out", (32, 32), output=True)), "x", "out", "a_ptr=x;out_ptr=out;BLOCK_SIZE=32", (1,), "zeros", 0, 0))
    add("triton_interleave", "triton", CORE, "test_interleave.kernel", ew, CORE, "z_ref = torch.stack([x, y], dim=-1).reshape(256)",
        spec((tensor("out", (256,), output=True, dtype="int32"),), "", "out", "Z=out;N=128", (1,), "interleave", 0, 0), note="生成映射，无输入 load；debug=False 保留原 decorator，未改函数体。")
    add("triton_ravel", "triton", STANDARD, "test_ravel.triton_ravel", ew, STANDARD, "torch.arange(0, 256, device=device)",
        spec((tensor("out", (256,), output=True, dtype="int32"),), "", "out", "out_ptr=out", (1,), "ravel", 0, 0))
    add("pytorch_sub", "pytorch", UTILS, "sub_kernel", ew, PTTEST, "shared sub_kernel; independent torch.sub mathematical specification",
        spec(vector, "x y", "out", "in_ptr0=x;in_ptr1=y;out_ptr=out;n_elements=x.numel();BLOCK_SIZE=128", (2,), "sub"),
        note="参考为独立 torch.sub 逻辑定义，非上游输出快照；共享测试 Kernel 在本测试文件未直接调用 sub，不能声称复用现成 sub 黄金测试。")
    add("pytorch_double", "pytorch", UTILS, "mul2_kernel", ew, PTTEST, "test_triton_kernel_with_views",
        spec((vector[0], vector[2]), "x", "out", "in_ptr0=x;out_ptr=out;n_elements=x.numel();BLOCK_SIZE=128", (2,), "double"))
    add("pytorch_double_strided", "pytorch", UTILS, "double_strided_kernel", ew, PTTEST, "test_triton_kernel_strided_input",
        spec((tensor("x", (16, 16)), tensor("out", (16, 16), output=True)), "x", "out", "in_ptr=x;out_ptr=out;in_y_stride=x.stride(0);out_y_stride=out.stride(0);X_BLOCK_SIZE=8;Y_BLOCK_SIZE=8", (2, 2), "double"),
        note="该函数为二维显式行 stride 映射，与一维 mul2 的 grid 和坐标不同；归入映射类，不用二维形状充作布局变换。")
    add("triton_broadcast", "triton", CORE, "test_broadcast.broadcast_kernel", feature, CORE, "np.broadcast_arrays(x, y)",
        spec((tensor("x", (32,64)), tensor("y", (64,)), tensor("out", (32,64), output=True)), "x y", "out", "x_ptr=x;y_ptr=y;y_broadcasted_ptr=out;M=32;N=64", (1,), "broadcast", 0, 0))
    add("triton_where_broadcast", "triton", CORE, "test_where_broadcast.where_kernel", feature, CORE, "z = np.where(mask, x, 0)",
        spec((tensor("x", (32,32)), tensor("mask", (32,), dtype="bool"), tensor("out", (32,32), output=True)), "x mask", "out", "cond_ptr=mask;a_ptr=x;out_ptr=out;BLOCK_SIZE=32", (1,), "where_broadcast", 0, 0))
    add("pytorch_masked_add", "pytorch", UTILS, "masked_add_kernel_with_bool_tensor", feature, PTTEST, "masked_add_kernel_with_bool_tensor; independent torch.where(mask, x+y, x)",
        spec((*vector, tensor("mask", (129,), dtype="bool")), "x y mask", "out", "in_ptr0=x;in_ptr1=y;mask_ptr=mask;out_ptr=out;n_elements=x.numel();BLOCK_SIZE=128", (2,), "masked_add"),
        note="多输入值掩码，不是访存 mask；真值来自独立 Eager 公式，未将数据相关条件授予 Fast。")
    for id, fn, bindings, grid in [
        ("liger_swiglu", "_swiglu_forward_kernel", "a_ptr=x;b_ptr=y;c_ptr=out;stride=x.stride(0);gate_multiplier=1.0;n_cols=256;BLOCK_SIZE=256", (32,)),
        ("liger_swiglu_tiled", "_swiglu_forward_kernel_tiled", "a_ptr=x;b_ptr=y;c_ptr=out;stride=x.stride(0);gate_multiplier=1.0;n_cols=256;BLOCK_SIZE=128", (32,2)),
    ]:
        add(id, "liger", "src/liger_kernel/ops/swiglu.py", fn, feature, "test/ops/test_swiglu.py", "_swiglu_ref",
            spec((tensor("x", (32,256)), tensor("y", (32,256)), tensor("out", (32,256), output=True)), "x y", "out", bindings, grid, "swiglu", 1e-5, 1e-5), "silu",
            "tiled 样本直接调用底层函数进行语义 smoke；不是上游 wide-row 选择阈值或性能实验。")
    add("liger_geglu", "liger", "src/liger_kernel/ops/geglu.py", "_geglu_tanh_forward_kernel", feature, "test/ops/test_geglu.py", "_geglu_ref",
        spec((tensor("x", (32,256)), tensor("y", (32,256)), tensor("out", (32,256), output=True)), "x y", "out", "a=x;b=y;c=out;stride=x.stride(0);n_cols=256;BLOCK_SIZE=256", (32,), "geglu", 1e-5, 1e-5))
    add("liger_fused_swiglu", "liger", "src/liger_kernel/ops/swiglu.py", "_swiglu_fused_gate_up_forward_kernel", feature, "test/ops/test_swiglu.py", "_swiglu_ref applied to gate/up halves",
        spec((tensor("x", (32,512)), tensor("out", (32,256), output=True)), "x", "out", "y_ptr=x;c_ptr=out;in_stride=x.stride(0);out_stride=out.stride(0);ffn_size=256;BLOCK_SIZE=256", (32,), "fused_swiglu", 1e-5, 1e-5), "silu",
        "独立参考复用 _swiglu_ref；输入拆半为本地审计适配，非原测试直接覆盖此入口。")
    add("flag_slice", "flag_gems", "src/flag_gems/ops/slice.py", "slice_kernel_2d", layout,
        "tests/test_slice.py", "torch.ops.aten.slice.Tensor(ref_inp, dim, start, end, step)",
        spec((tensor("x", (16,32)), tensor("out", (16,15), output=True)), "x", "out",
             "input_ptr=x;output_ptr=out;dim=1;start=2;step=2;input_stride_0=x.stride(0);input_stride_1=x.stride(1);output_size_0=16;output_size_1=15;BLOCK_SIZE=16",
             (16,1), "slice2d", 0, 0),
        note="输出坐标和 stride 均为元素单位；直接传 Tensor 指针，未重复叠加 storage_offset。")
    add("triton_permute", "triton", CORE, "test_permute.kernel", layout, CORE, "z_ref = x.transpose(*perm)",
        spec((tensor("x", (64,64)), tensor("out", (64,64), output=True)), "x", "out", "X=x;stride_xm=x.stride(0);stride_xn=x.stride(1);Z=out;stride_zm=out.stride(1);stride_zn=out.stride(0);BLOCK_M=64;BLOCK_N=64", (1,1), "transpose", 0, 0))
    add("triton_trans2d", "triton", CORE, "test_trans_2d.kernel", layout, CORE, "expected = torch.permute(input, perm)",
        spec((tensor("x", (16,16), dtype="int32", fill="arange"), tensor("out", (16,16), output=True, dtype="int32")), "x", "out", "In=x;Out=out;in_shape1=16;in_shape2=16;ou_shape1=16;ou_shape2=16;trans1=1;trans2=0", (1,), "transpose", 0, 0))
    add("triton_flip", "triton", STANDARD, "test_flip.flip_kernel", layout, STANDARD, "y = torch.flip(x, ...)",
        spec((tensor("x", (8,2,256)), tensor("out", (8,2,256), output=True)), "x", "out", "X=x;Z=out;M=8;N=2;K=256;dim=2", (1,), "flip", 0, 0))
    add("triton_pair_flip", "triton", STANDARD, "test_flip_inf.triton_flip_kernel", layout, STANDARD, "expect = x.reshape(-1, 8, 2).flip(-1).reshape(-1, 16)",
        spec((tensor("x", (1,16), fill="arange_inf"), tensor("out", (1,16), output=True)), "x", "out", "out_ptr=out;x_ptr=x;N=16", (1,), "pair_flip", 0, 0), note="smoke 采用 arange 且包含末元素 inf，与上游复现一致；不把历史 Issue 分类为布局漏洞。")
    add("triton_split", "triton", CORE, "test_split.kernel", layout, CORE, "z1_ref, z2_ref = (x[:, 0], x[:, 1])",
        spec((tensor("x", (128,2), dtype="int32", fill="arange"), tensor("out", (128,), output=True, dtype="int32"), tensor("out2", (128,), output=True, dtype="int32")), "x", "out out2", "X=x;Z1=out;Z2=out2;N=256", (1,), "split", 0, 0))
    add("triton_softmax", "triton", STANDARD, "test_softmax.softmax_kernel", reduction, STANDARD, "torch.softmax(x, dim=dim)",
        spec((tensor("x", (128,)), tensor("out", (128,), output=True)), "x", "out", "X=x;Z=out;numel=128;shape=(128,);dim=0;ieee_rounding=False", (1,), "softmax"),
        note="采用上游参数矩阵中的一维 GPU 可执行项；二维 axis 复现仅标为当前 Triton 版本兼容缺口，不改写函数体。")
    add("liger_softmax", "liger", "src/liger_kernel/ops/softmax.py", "_softmax_single_block_forward_kernel", reduction, "test/ops/test_softmax.py", "_softmax_ref",
        spec((tensor("x", (32,256)), tensor("out", (32,256), output=True)), "x", "out", "Y_ptr=out;Y_row_stride=out.stride(0);X_ptr=x;X_row_stride=x.stride(0);n_cols=256;BLOCK_SIZE=256", (32,), "softmax", 1e-5, 1e-5))
    norm = (tensor("x", (8,80)), tensor("w", (80,)), tensor("b", (80,)), tensor("out", (8,80), output=True), tensor("mean", (8,), output=True), tensor("rstd", (8,), output=True))
    add("liger_layernorm", "liger", "src/liger_kernel/ops/layer_norm.py", "_layer_norm_forward_kernel", reduction, "test/ops/test_layer_norm.py", "pytorch_reference_layer_norm",
        spec(norm, "x w b", "out mean rstd", "Y_ptr=out;Y_row_stride=out.stride(0);X_ptr=x;X_row_stride=x.stride(0);W_ptr=w;W_row_stride=1;B_ptr=b;B_row_stride=1;Mean_ptr=mean;Mean_row_stride=1;RSTD_ptr=rstd;RSTD_row_stride=1;n_cols=80;eps=1e-5;BLOCK_SIZE=128", (8,), "layernorm", 1e-4, 1e-5), note="辅助 mean/rstd 另由 PyTorch 行统计量核对；小 shape 为 e1 有界适配，不声称运行全部上游参数矩阵。")
    add("unsloth_layernorm", "unsloth", "unsloth/kernels/layernorm.py", "layernorm_forward", reduction, "tests/test_layernorm_gradient_layout.py", "expected = torch.nn.functional.layer_norm",
        spec(norm, "x w b", "out rstd mean", "Y=out;Y_row_stride=out.stride(0);X=x;X_row_stride=x.stride(0);W=w;b=b;r=rstd;mu=mean;n_cols=80;eps=1e-5;BLOCK_SIZE=128", (8,), "layernorm", 1e-2, 1e-3))
    rms = (norm[0], norm[1], norm[3], norm[5])
    rms_bind = "Y_ptr=out;Y_row_stride=out.stride(0);X_ptr=x;X_row_stride=x.stride(0);W_ptr=w;W_row_stride=1;RSTD_ptr=rstd;RSTD_row_stride=1;"
    for id, fn, extra, grid in [("liger_rmsnorm", "_rms_norm_forward_kernel", "", (8,)), ("liger_block_rmsnorm", "_block_rms_norm_forward_kernel", "n_rows=8;", (2,))]:
        bind = rms_bind + extra + "n_cols=80;eps=1e-6;offset=0.0;casting_mode=0;elementwise_affine=True;BLOCK_SIZE=128" + (";BLOCK_ROW=4" if extra else "")
        add(id, "liger", "src/liger_kernel/ops/rms_norm.py", fn, reduction, "test/ops/test_rms_norm.py", "pytorch_reference_rms_norm",
            spec(rms, "x w", "out rstd", bind, grid, "rmsnorm", 1e-4, 1e-4), note="上游明确继承 Unsloth RMSNorm；记录共同谱系，不把两个项目名当作独立算法证据。")
    for id, fn, ref in [("unsloth_rmsnorm", "_rms_layernorm_forward", "rmsnorm"), ("unsloth_gemma_rmsnorm", "_gemma_rms_layernorm_forward", "gemma_rmsnorm")]:
        add(id, "unsloth", "unsloth/kernels/rms_layernorm.py", fn, reduction, "tests/test_rmsnorm_gradient_layout.py", "torch.nn.functional.rms_norm / Gemma weight offset",
            spec(rms, "x w", "out rstd", "Y=out;Y_row_stride=out.stride(0);X=x;X_row_stride=x.stride(0);W=w;W_row_stride=1;r=rstd;r_row_stride=1;n_cols=80;eps=1e-6;BLOCK_SIZE=128", (8,), ref, 1e-2, 1e-3),
            note="辅助 rstd 用独立行统计量；Gemma 的 weight+1 与 cast 语义单列审计，共同 RMS 谱系不隐去。")
    challenges = [
        ("triton_signed_neg", "triton", CORE, "_neg_signed_zero_kernel", ew),
        ("triton_cat", "triton", CORE, "test_cat.kernel", layout),
        ("triton_tutorial_softmax", "triton", "python/tutorials/02-fused-softmax.py", "softmax_kernel", reduction),
        ("triton_tutorial_layernorm", "triton", "python/tutorials/05-layer-norm.py", "_layer_norm_fwd_fused", reduction),
        ("triton_gather", "triton", CORE, "gather_test_kernel", "indirect_index"),
        ("triton_trans4d", "triton", CORE, "test_trans_4d.kernel", "tensor_descriptor"),
        ("triton_reshape_template", "triton", CORE, "test_reshape.kernel", "source_template"),
        ("liger_swiglu_backward", "liger", "src/liger_kernel/ops/swiglu.py", "_swiglu_backward_kernel", "inplace_multi_store"),
        ("unsloth_swiglu", "unsloth", "unsloth/kernels/swiglu.py", "_fg_kernel", feature),
        ("unsloth_rope", "unsloth", "unsloth/kernels/rope_embedding.py", "_rope_embedding", "inplace_rope"),
        ("flag_cat", "flag_gems", "src/flag_gems/ops/cat.py", "cat_copy_func_kernel_4", layout),
        ("flag_complex_transpose", "flag_gems", "src/flag_gems/ops/transpose_copy.py", "_complex_transpose_tiled_kernel", "complex_static_loop"),
    ]
    for id, source, file, fn, family in challenges:
        compatibility = {
            "triton_signed_neg": "上游仅以 pytest.mark.interpreter 覆盖；当前 GPU smoke 数值相等但 signbit 不一致，故降为挑战。",
            "triton_cat": "当前 GPU 后端拒绝 can_reorder=False；True 会改变逐元素顺序，故不冒称 torch.cat 语义完整。",
        }.get(id, "")
        note = "挑战仅固定来源和拒绝边界，不进入正向分母、不重写参考。"
        add(id, source, file, fn, family, "", "", None, note=note + compatibility)
    return tuple(rows)
