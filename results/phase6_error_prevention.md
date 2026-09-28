# 第六阶段受限布局违约拦截

生成时间：2026-09-25T10:53:13.274477+08:00
逐例阻止：4/4。

| 输入 | 原始 Fast | 旧 Guard | 新分派 | 阻止且结果正确 |
| --- | --- | --- | --- | --- |
| A transpose 7×11 | numeric_mismatch | correct / PyTorch Fallback | correct / PyTorch Fallback | True |
| A transpose 5×13 | numeric_mismatch | correct / PyTorch Fallback | correct / PyTorch Fallback | True |
| D x_strided 7×129 | numeric_mismatch | correct / PyTorch Fallback | correct / PyTorch Fallback | True |
| D y_strided 7×129 | numeric_mismatch | correct / PyTorch Fallback | correct / PyTorch Fallback | True |

D 来自 Triton 官方向量加法教程的本地适配；违约风险是输入布局改变后的可复现模式，不声称上游教程原始 wrapper 存在历史 bug。
只覆盖 A 和 D 的四个已知错读输入；异常、超时和未证实内存错误不能合并为已阻止错误。
