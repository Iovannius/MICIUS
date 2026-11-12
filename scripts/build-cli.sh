#!/usr/bin/env bash
set -euo pipefail
package=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PATH="${MICIUS_CLANG64_BIN:-/c/msys64/clang64/bin}:$PATH"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUTF8=1
command -v python >/dev/null
exec python -B "$package/scripts/build_cli.py" "$@"
