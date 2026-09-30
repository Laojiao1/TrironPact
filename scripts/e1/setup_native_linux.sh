#!/usr/bin/env bash
set -euo pipefail

# 仅为用户取得的原生 Linux 候选机创建隔离环境；不会预订资源或运行性能实验。
if [[ "$(uname -s)" != "Linux" ]] || grep -qi microsoft /proc/version; then
  echo "Unsupported: this setup requires native Linux, not WSL" >&2
  exit 2
fi

if [[ $# -ne 1 ]]; then
  echo "usage: bash scripts/e1/setup_native_linux.sh <venv-path>" >&2
  exit 2
fi

if [[ ! -f "scripts/e1/native_smoke.py" ]]; then
  echo "Unknown: run this script from the TritonPact repository root" >&2
  exit 2
fi

environment_path="$1"
python3 -m venv "$environment_path"
"$environment_path/bin/python" -m pip install --upgrade "pip==25.2"
"$environment_path/bin/python" -m pip install --index-url https://download.pytorch.org/whl/cu128 "torch==2.11.0"
"$environment_path/bin/python" -m pip install "triton==3.6.0" "pytest==8.4.2"
"$environment_path/bin/python" scripts/e1/native_smoke.py
