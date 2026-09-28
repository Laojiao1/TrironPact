"""固定 vLLM 源码挑战样例，仅供离线 AST 拒绝审计。

原始文件：vllm/v1/worker/gpu/input_batch.py
提交：e6c07ea5763bada5da789ed437851ee1526fd5a8
函数：_expand_idx_mapping_kernel（原样保留函数体和签名）
许可：Apache-2.0，见 bench/THIRD_PARTY_NOTICES.md。
"""

import triton
import triton.language as tl


@triton.jit
def _expand_idx_mapping_kernel(
    idx_mapping_ptr,
    expanded_idx_mapping_ptr,
    expanded_local_pos_ptr,
    cu_num_logits_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0)
    start_idx = tl.load(cu_num_logits_ptr + req_idx)
    end_idx = tl.load(cu_num_logits_ptr + req_idx + 1)
    num_tokens = end_idx - start_idx

    block = tl.arange(0, BLOCK_SIZE)
    mask = block < num_tokens
    req_state_idx = tl.load(idx_mapping_ptr + req_idx)
    tl.store(expanded_idx_mapping_ptr + start_idx + block, req_state_idx, mask=mask)
    tl.store(expanded_local_pos_ptr + start_idx + block, block, mask=mask)
