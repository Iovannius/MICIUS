#!/usr/bin/env bash
set -euo pipefail
case "${1:-all}" in
  all|prepare|run|chat) ;;
  --help) echo 'Usage: run_m2.sh {all|prepare|run RESULT_DIRECTORY|chat {smol|qwen} [INT8|F64]}'; exit 0 ;;
  *) echo 'Expected all, prepare, run or chat' >&2; exit 2 ;;
esac
package=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PATH="${MICIUS_CLANG64_BIN:-/c/msys64/clang64/bin}:$PATH"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUTF8=1
command -v python >/dev/null
if [[ $# == 0 ]]; then set -- all; fi
exec python -B "$package/scripts/m2_run.py" "$@"
