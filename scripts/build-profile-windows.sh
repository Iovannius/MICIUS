#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
case "${1:-}" in
    int8-release-windows) build=build-int8-release ;;
    f64-reference-windows) build=build-f64-reference ;;
    *) echo 'Expected int8-release-windows or f64-reference-windows.' >&2; exit 2 ;;
esac
profile=$1
clang_bin=${MICIUS_CLANG64_BIN:-/c/msys64/clang64/bin}
if [[ -d "$clang_bin" ]]; then export PATH="$clang_bin:$PATH"; fi
for tool in python clang++ cmake ctest ninja sha256sum; do command -v "$tool" >/dev/null; done
case "$(clang++ -dumpmachine)" in
    x86_64-*-windows-gnu|x86_64-w64-mingw32) ;;
    *) echo 'Use Windows x64 Clang64, or set MICIUS_CLANG64_BIN.' >&2; exit 1 ;;
esac
export PYTHONDONTWRITEBYTECODE=1
python -B scripts/verify_release.py
if [[ -e "$build" ]]; then
    echo "Preserve or rename $build before this clean build." >&2
    exit 1
fi
mkdir -p logs
reports=$(mktemp -d "logs/${profile}.XXXXXX")
clang++ --version > "$reports/compiler.txt"
cmake --version > "$reports/cmake.txt"
ninja --version > "$reports/ninja.txt"
cmake --preset "$profile" 2>&1 | tee "$reports/configure.txt"
cmake --build --preset "$profile" 2>&1 | tee "$reports/build.txt"
ctest --preset "$profile" --output-on-failure 2>&1 | tee "$reports/ctest.txt"
python -B scripts/verify_release.py | tee "$reports/source-check.txt"
sha256sum "$build/bin/micius_cli.exe" "$build/bin/micius_sandbox.exe" > "$reports/executables.sha256"
printf 'profile=%s\ncli=%s/bin/micius_cli.exe\nbuild_reports=%s\n' "$profile" "$build" "$reports"
