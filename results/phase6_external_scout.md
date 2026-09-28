# 第六阶段第三方源码选型摸底

时间：2026-09-25T11:05:38.374388+08:00

来源：https://github.com/linkedin/Liger-Kernel/blob/40a9d8a61bdc546f4e70220619d4db9760fa80b1/src/liger_kernel/ops/swiglu.py

提交：`40a9d8a61bdc546f4e70220619d4db9760fa80b1`；文件 SHA-256：`7397029e8fefac0de42e5eeafcbbac147644a91c357bd30b9ba019fa5df3ce0a`。

函数：`_swiglu_forward_kernel`，第 57 行，函数 SHA-256：`021883fbe4e673a3845a84ff4d686905bd8850c7fb38815d407ef4f425d6685f`。

结果：**Unsupported**；不支持第 8 行的控制流或语句。

纯语法摸底；不导入、不运行、未核对 wrapper 与逻辑语义；AST Supported 也不授予 Fast。
