#!/usr/bin/env python3
"""Verify the delivered runtime inputs; leave local edits untouched on failure."""
import hashlib
import json
from pathlib import Path
import sys

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root / "docs/f64-closeout-evidence/runtime-source.json").read_text(encoding="utf-8"))
expected = manifest["source_files"]
errors = []
for relative, wanted in expected.items():
    path = root / relative
    if not path.is_file():
        errors.append("missing: " + relative)
    elif hashlib.sha256(path.read_bytes()).hexdigest() != wanted:
        errors.append("modified: " + relative)

# CMake uses GLOB_RECURSE. Leftover experiment sources can silently enter a
# nominal baseline build, so check the code tree as well as individual hashes.
extensions = {".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".hxx", ".inl", ".inc"}
for folder in ("src", "include", "tests"):
    for path in sorted((root / folder).rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_file() and path.suffix.lower() in extensions and relative not in expected:
            errors.append("unexpected code file: " + relative)

if errors:
    print("source_check=FAIL\n" + "\n".join(errors), file=sys.stderr)
    print("Keep local edits. Reconcile them with runtime-changes.json before using this profile.", file=sys.stderr)
    sys.exit(1)
print("source_check=PASS")
print("verified_runtime_inputs=" + str(len(expected)))
