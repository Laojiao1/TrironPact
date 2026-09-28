# 第六阶段第三方来源记录

## Triton

- 固定提交：`f394c9bb86b2eaca08b0e11c4cd397d19a56c693`
- 原始文件：`python/tutorials/01-vector-add.py`
- 许可：MIT，完整文本见 `bench/TRITON_LICENSE.txt`
- 本地文件：`scenarios/external_add.py`，仅保留 Kernel 和最小启动函数

## vLLM

- 固定提交：`e6c07ea5763bada5da789ed437851ee1526fd5a8`
- 原始文件：`vllm/v1/worker/gpu/input_batch.py`
- 原文件 SHA-256：`14656c08545453414c2b89ba400cb20c649e9289e1dc0cf1540b83ab1957ad59`
- 许可：Apache-2.0；固定提交许可证 SHA-256：`c71d239df91726fc519c6eb72d318ec65820627232b2f796219e87dcf35d0ab4`
- 本地文件：`scenarios/external_vllm.py`，函数体和签名原样保留，仅增加来源说明与导入
- 用途：离线挑战集。该函数含无显式 mask 的标量 load 和两个 store，预期被当前 Access IR 明确拒绝，不进入 Fast。
