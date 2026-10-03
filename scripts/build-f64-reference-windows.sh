#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

for tool in python clang++ cmake ctest ninja sha256sum; do
    command -v "$tool" >/dev/null
done
case "$(clang++ -dumpmachine)" in
    x86_64-*-windows-gnu|x86_64-w64-mingw32) ;;
    *) echo 'This profile requires the Windows x64 Clang64 toolchain.' >&2; exit 1 ;;
esac
python scripts/verify-f64-reference.py
if [ -e build-f64-reference ]; then
    echo 'Preserve or rename build-f64-reference before this clean baseline build.' >&2
    exit 1
fi

mkdir -p logs
REPORTS=$(mktemp -d logs/f64-reference-build.XXXXXX)
echo "build_reports=$REPORTS"
clang++ --version | tee "$REPORTS/compiler.txt"
clang++ -dumpmachine | tee "$REPORTS/compiler-target.txt"
cmake --version > "$REPORTS/cmake.txt"
ninja --version > "$REPORTS/ninja.txt"
cmake --preset f64-reference-windows 2>&1 | tee "$REPORTS/configure.txt"
cmake --build --preset f64-reference-windows 2>&1 | tee "$REPORTS/build.txt"
ctest --preset f64-reference-windows 2>&1 | tee "$REPORTS/ctest.txt"
python scripts/verify-f64-reference.py | tee "$REPORTS/source-check.txt"
sha256sum build-f64-reference/bin/micius_cli.exe \
    build-f64-reference/bin/micius_sandbox.exe > "$REPORTS/executables.sha256"
echo 'f64_reference_build_and_ctest=PASS'
echo 'cli=C:/Projects/MICIUS/build-f64-reference/bin/micius_cli.exe (default extraction path)'
