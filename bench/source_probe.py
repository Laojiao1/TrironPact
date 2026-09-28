"""对固定版第三方源码做纯 AST 预筛；不导入或执行第三方项目。"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

from bench.inventory import function_source
from pact.access_ir import parse_access_ir


def probe(path: Path, function: str, pointers: dict[str, str], *, source_url: str, revision: str) -> dict:
    raw = path.read_bytes()
    source, line = function_source(path, function)
    result = parse_access_ir(source, pointers, {name: 4 for name in pointers})
    return {"generated_at": datetime.now().astimezone().isoformat(), "source_url": source_url,
            "revision": revision, "file_sha256": hashlib.sha256(raw).hexdigest(), "function": function,
            "line": line, "function_sha256": hashlib.sha256(source.encode("utf-8")).hexdigest(),
            "parse_status": result.status, "parse_reason": result.reason,
            "accesses_before_rejection": len(result.accesses),
            "scope": "纯语法摸底；不导入、不运行、未核对 wrapper 与逻辑语义；AST Supported 也不授予 Fast。"}


def render(data: dict) -> str:
    return ("# 第六阶段第三方源码选型摸底\n\n"
            f"时间：{data['generated_at']}\n\n"
            f"来源：{data['source_url']}\n\n"
            f"提交：`{data['revision']}`；文件 SHA-256：`{data['file_sha256']}`。\n\n"
            f"函数：`{data['function']}`，第 {data['line']} 行，函数 SHA-256：`{data['function_sha256']}`。\n\n"
            f"结果：**{data['parse_status']}**；{data['parse_reason']}。\n\n{data['scope']}\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--function", required=True)
    parser.add_argument("--pointers", required=True, help="如 a_ptr=X,b_ptr=Y,c_ptr=OUT")
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        pointers = dict(part.split("=", 1) for part in args.pointers.split(","))
    except ValueError:
        parser.error("指针绑定应为 name=tensor 的逗号分隔列表")
    data = probe(args.input, args.function, pointers, source_url=args.source_url, revision=args.revision)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(data), encoding="utf-8")
    args.output.with_suffix(".json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(data["parse_status"], data["parse_reason"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
