#!/usr/bin/env bash
# 构建 C++ 引擎：共享库（供 Python ctypes）、自测与基准程序
# 用法：scripts/build.sh [debug]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/build"
mkdir -p "$OUT"
CXX="${CXX:-g++}"
if [[ "${1:-}" == "debug" ]]; then
  OPT=(-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer)
else
  OPT=(-O3 -march=native -DNDEBUG)
fi
FLAGS=(-std=c++17 -Wall -Wextra -fPIC -pthread -I"$ROOT/engine/include" "${OPT[@]}")
SRC=("$ROOT"/engine/src/{board,rules,game,encode,agents,selfplay}.cpp)

"$CXX" "${FLAGS[@]}" -shared -fvisibility=hidden "${SRC[@]}" "$ROOT/engine/src/capi.cpp" -o "$OUT/libjunqi.so"
"$CXX" "${FLAGS[@]}" "${SRC[@]}" "$ROOT/engine/tests/test_engine.cpp" -o "$OUT/test_engine"
"$CXX" "${FLAGS[@]}" "${SRC[@]}" "$ROOT/engine/tests/bench.cpp" -o "$OUT/bench"
mkdir -p "$ROOT/junqi/_lib"
cp "$OUT/libjunqi.so" "$ROOT/junqi/_lib/"
echo "built: $OUT/libjunqi.so test_engine bench"
