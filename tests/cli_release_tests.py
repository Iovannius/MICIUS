"""Exercise the shipped CLI on packed, untied and whole-model fallback layouts."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

from reference.make_fixture import make_fixture


def check(executable, backend, golden):
    expected = json.loads((Path(__file__).parent / "reference/cli-traces.json").read_text())
    with tempfile.TemporaryDirectory(dir=Path.cwd(), prefix="cli-fixture-") as directory:
        work = Path(directory)
        for name, head, heads, tied, vocab in (
            ("gqa3", 64, 3, True, 128),
            ("untied", 32, 2, False, 128),
            ("tail", 32, 2, True, 127),
        ):
            fixture = make_fixture(work / "tiny.gguf", context=256, head_dim=head, heads=heads,
                                   kv_heads=1, q8_embedding=tied, tied=tied, vocab=vocab)
            record = expected[backend][name]
            command = [executable, "--model-benchmark", "--model", str(fixture), "--token", "42"]
            run = subprocess.run(command + ["--steps", "4"], cwd=work, capture_output=True, check=True)
            if f"gemv_backend={record['reported_kernel']}".encode() not in run.stdout.splitlines():
                raise ValueError(f"{name}: unexpected model dispatch")
            trace = work / "out.trace"
            subprocess.run(command + ["--steps", "160", "--pw2-parity", str(trace)],
                           cwd=work, capture_output=True, check=True)
            data = trace.read_bytes()
            if len(data) != 4 + 160 * (8 + 4 * vocab):
                raise ValueError(f"{name}: incomplete prediction/logits trace")
            if golden and hashlib.sha256(data).hexdigest() != record["trace_sha256"]:
                raise ValueError(f"{name}: output differs from the pinned Windows reference")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("executable", type=Path)
    parser.add_argument("backend", choices=("INT8", "F64"))
    parser.add_argument("--golden", action="store_true")
    args = parser.parse_args()
    check(args.executable.resolve(), args.backend, args.golden)
