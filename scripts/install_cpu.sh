#!/usr/bin/env bash
# 安装 CPU 版 PyTorch、MJLab，以及本仓库的外肢体抓取任务。
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python3}"
echo "使用解释器：$("$PYTHON" -c 'import sys; print(sys.executable)')"

"$PYTHON" -m pip install -U pip
echo "正在安装 CPU 版 PyTorch 和 MJLab……"
# 把 CPU 轮子放在主索引，避免 MJLab 再拉一套 CUDA 版 PyTorch。
"$PYTHON" -m pip install mjlab \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple
echo "正在以可编辑模式安装本仓库……"
"$PYTHON" -m pip install -e "$ROOT"

echo "安装完成。下一步：python scripts/smoke_test.py"
