"""Check the v0.7 source tree. Git LF and Windows CRLF have the same identity."""
import hashlib
import json
from pathlib import Path


def verify():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "docs/evidence/v0.7-source.json").read_text(encoding="utf-8"))
    expected = manifest["files"]
    names = {"CMakeLists.txt", "CMakePresets.json"}
    for folder in ("src", "include", "scripts", "tests"):
        for path in (root / folder).rglob("*"):
            if path.is_symlink():
                raise SystemExit(f"Unexpected symbolic link: {path.relative_to(root)}")
            if path.suffix == ".pyc" and "__pycache__" in path.parts:
                continue
            if path.is_file():
                names.add(path.relative_to(root).as_posix())
    errors = []
    for name in sorted(names | expected.keys()):
        path = root / name
        if name not in expected:
            errors.append("unexpected: " + name)
        elif name not in names or not path.is_file():
            errors.append("missing: " + name)
        elif hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest() != expected[name]:
            errors.append("modified: " + name)
    if errors:
        raise SystemExit("v0.7 source check failed\n" + "\n".join(errors))
    print(f"v0.7 source check passed ({len(expected)} source, test and build files)")


if __name__ == "__main__":
    verify()
