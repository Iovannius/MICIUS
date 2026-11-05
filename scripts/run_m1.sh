#!/usr/bin/env bash
set -euo pipefail
case "${1:-all}" in
    all|prepare|run) ;;
    --help) echo 'Usage: bash scripts/run_m1.sh {all|prepare|run RESULT_DIRECTORY}'; exit 0 ;;
    *) echo 'Expected all, prepare, or run RESULT_DIRECTORY' >&2; exit 2 ;;
esac
package=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
clang_bin=${MICIUS_CLANG64_BIN:-/c/msys64/clang64/bin}
export PATH="$clang_bin:$PATH"
export PYTHONDONTWRITEBYTECODE=1
command -v python >/dev/null
if [[ $# == 0 ]]; then set -- all; fi
exec python -B "$package/scripts/m1_run.py" "$@"
