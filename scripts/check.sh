#!/usr/bin/env bash
# 全量自检：构建 → C++ 自测 → 规则回归 → Python 测试（含泄漏测试）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python3}"
[[ -x "$ROOT/.venv/bin/python" ]] && PY="$ROOT/.venv/bin/python"
"$ROOT/scripts/build.sh"
"$ROOT/build/test_engine"
"$PY" "$ROOT/tests/test_regression.py" | tail -1
"$PY" "$ROOT/tests/test_python.py" | tail -1
