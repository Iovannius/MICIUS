"""Verify the v0.7.3 release, including complete source-directory file sets."""
import hashlib
import json
from pathlib import Path

MANIFEST = Path(__file__).resolve().parents[1] / "RELEASE-MANIFEST.json"


def verify():
    root = MANIFEST.parent
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if manifest["release"] != "0.7.3":
        raise SystemExit("Expected the MICIUS v0.7.3 release manifest")
    expected = manifest["files"]
    names = {"CMakeLists.txt", "CMakePresets.json", ".gitattributes", ".gitignore", "LICENSE", "README.md"}
    for folder in ("src","include","scripts","tests","docs","cmake","examples","validation"):
        if (root / folder).is_symlink():
            raise SystemExit(f"Unexpected symbolic link: {folder}")
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
        elif name not in names or path.is_symlink() or not path.is_file():
            errors.append("missing: " + name)
        else:
            data = path.read_bytes()
            if path.suffix == ".sh" and b"\r" in data:
                errors.append("Bash script requires LF: " + name)
            if path.suffix != ".zlib":
                data = data.replace(b"\r\n", b"\n")
            if hashlib.sha256(data).hexdigest() != expected[name]:
                errors.append("modified: " + name)
    if errors:
        raise SystemExit("v0.7.3 source check failed\n" + "\n".join(errors))
    print(f"v0.7.3 source check passed ({len(expected)} distribution files)")


if __name__ == "__main__":
    verify()
