"""固定并保存上游原始资产；仅下载源码与许可，不导入远程项目。

首次解析指定 Git revision，之后只使用 sources.json 中的完整提交。
网络失败保留错误，不以空文件或未固定 main 代替来源证据。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPOS = {
    "triton": ("triton-lang/triton", "f394c9bb86b2eaca08b0e11c4cd397d19a56c693"),
    "liger": ("linkedin/Liger-Kernel", "40a9d8a61bdc546f4e70220619d4db9760fa80b1"),
    "unsloth": ("unslothai/unsloth", "main"),
    "flag_gems": ("FlagOpen/FlagGems", "master"),
    "pytorch": ("pytorch/pytorch", "main"),
}


def fetch(url: str) -> bytes:
    """读取公开 HTTPS 资产；超时和 HTTP 错误向调用者传播。"""
    request = urllib.request.Request(url, headers={"User-Agent": "TritonPact-e1-source-audit"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pin", action="store_true")
    parser.add_argument("--source", choices=REPOS)
    parser.add_argument("--files", nargs="+")
    args = parser.parse_args()
    manifest = ROOT / "sources.json"
    data = json.loads(manifest.read_text()) if manifest.exists() else {}
    if args.pin:
        for key, (repo, revision) in REPOS.items():
            if key in data:
                continue
            commit = json.loads(fetch(f"https://api.github.com/repos/{repo}/commits/{revision}"))
            data[key] = {"repository": repo, "revision": commit["sha"], "commit_date": commit["commit"]["committer"]["date"], "assets": {}}
            manifest.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
            print(key, data[key]["revision"], flush=True)
    if args.source and args.files:
        source = data[args.source]
        for path in args.files:
            target = ROOT / "upstream" / args.source / path
            url = f"https://raw.githubusercontent.com/{source['repository']}/{source['revision']}/{path}"
            raw = fetch(url)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            source["assets"][path] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "url": url}
            current = json.loads(manifest.read_text())
            current[args.source] = source
            manifest.write_text(json.dumps(current, indent=2) + "\n", encoding="utf-8")
            print(args.source, path, len(raw), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
