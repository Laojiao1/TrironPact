"""构造各种特殊物理布局的张量（转置、padded、offset、strided、单行/列等）"""

import torch

def make_case(name: str, layout: str, m: int = 7, n: int = 11) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    if name in ("A", "A2"):
        if layout == "contiguous":
            return torch.arange(m * n, device="cuda", dtype=torch.float32).reshape(m, n)
        if layout == "transpose":
            return torch.arange(m * n, device="cuda", dtype=torch.float32).reshape(n, m).T
        if layout == "padded":
            base = torch.arange(m * (n + 3), device="cuda", dtype=torch.float32).reshape(m, n + 3)
            return base[:, :n]
        if layout == "unknown_dtype":
            return torch.arange(m * n, device="cuda", dtype=torch.float16).reshape(m, n)
        if name == "A" and layout == "single_row" and m == 1 and n > 1:
            base = torch.arange(n + 3, device="cuda", dtype=torch.float32).reshape(1, n + 3)
            return base[:, :n]
        if name == "A" and layout == "single_col" and n == 1 and m > 1:
            base = torch.arange(m, device="cuda", dtype=torch.float32)
            return torch.as_strided(base, (m, 1), (1, 3))
    elif name == "B":
        if layout == "contiguous":
            return torch.arange(n, device="cuda", dtype=torch.float32)
        if layout == "offset":
            return torch.arange(n + 1, device="cuda", dtype=torch.float32)[1:]
        if layout == "strided":
            return torch.arange(n * 2, device="cuda", dtype=torch.float32)[::2]
        if layout == "unknown_dtype":
            return torch.arange(n, device="cuda", dtype=torch.float16)
    elif name == "D":
        dtype = torch.float16 if layout == "unknown_dtype" else torch.float32
        x = torch.arange(n, device="cuda", dtype=dtype)
        y = torch.arange(n, device="cuda", dtype=dtype) * 10
        if layout in ("contiguous", "unknown_dtype"):
            return x, y
        if layout == "x_strided":
            return torch.arange(n * 2, device="cuda", dtype=dtype)[::2], y
        if layout == "y_strided":
            return x, torch.arange(n * 2, device="cuda", dtype=dtype)[::2]
        if layout == "x_offset":
            return torch.arange(n + 1, device="cuda", dtype=dtype)[1:], y
    raise ValueError(f"未知输入：{name}/{layout}")


def reference(x: torch.Tensor, y: torch.Tensor | None = None, *, case: str = "") -> torch.Tensor:
    return x + y if y is not None else x + 1
