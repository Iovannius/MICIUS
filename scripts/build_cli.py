"""Build and test the release CLI, then publish a self-contained runtime folder.

Windows x64 / MSYS2 Clang64 only. No model inference or Git operations.
Existing build/output directories are preserved and must be renamed by the user.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from verify_release import MANIFEST, verify

ROOT = MANIFEST.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    if os.name != 'nt':
        raise RuntimeError('Use Git Bash or MSYS2 Bash on Windows x64 with Clang64.')
    tools = {}
    for name in ('clang++', 'cmake', 'ctest', 'ninja', 'llvm-objdump', 'pkg-config'):
        path = shutil.which(name)
        if not path:
            raise RuntimeError('Required Clang64 tool not found: ' + name)
        tools[name] = str(Path(path).resolve())
    triple = subprocess.check_output([tools['clang++'], '-dumpmachine'], text=True).strip()
    if not re.fullmatch(r'x86_64-(?:[^-]+-windows-gnu|w64-mingw32)', triple):
        raise RuntimeError('Expected Windows x64 Clang64; got ' + triple)
    verify()
    build = ROOT / 'build-cli-int8'
    target = ROOT / 'micius_cli'
    for path in (build, target):
        if path.exists() or path.is_symlink():
            raise RuntimeError('Preserve or rename the existing directory before rebuilding: ' + str(path))
    build.mkdir()
    logs = build / 'delivery-logs'
    logs.mkdir()
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1')

    def run(name, args, cwd=ROOT, env=None):
        print(name, flush=True)
        with (logs / (name + '.log')).open('wb') as log:
            child = subprocess.Popen(list(map(str, args)), cwd=cwd, env=env,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            try:
                for line in child.stdout:
                    log.write(line)
                    sys.stdout.buffer.write(line)
                    sys.stdout.buffer.flush()
                code = child.wait()
            except BaseException:
                child.terminate()
                child.wait()
                raise
        if code:
            raise RuntimeError(name + ' failed; see ' + str(logs))

    run('configure', [tools['cmake'], '--preset', 'int8-release-windows', '-B', build,
                      '-DPython3_EXECUTABLE=' + sys.executable])
    run('build', [tools['cmake'], '--build', build, '--parallel', '8'])
    run('ctest', [tools['ctest'], '--test-dir', build, '--output-on-failure', '--no-tests=error'])
    verify()
    exe = build / 'bin/micius_cli.exe'
    if not exe.is_file() or exe.stem != target.name:
        raise RuntimeError('Unexpected CLI filename; no runtime directory published.')

    # Only Windows DLLs are supplied by the OS. Resolve every other import from
    # this compiler's bin directory, never from the current working directory.
    system_dlls = {'kernel32.dll', 'user32.dll', 'advapi32.dll', 'shell32.dll',
                   'ole32.dll', 'oleaut32.dll', 'bcrypt.dll', 'ntdll.dll',
                   'ws2_32.dll', 'ucrtbase.dll', 'msvcrt.dll', 'gdi32.dll'}
    runtime = {exe.name: exe}
    queue = [exe]
    compiler_bin = Path(tools['clang++']).parent
    while queue:
        item = queue.pop()
        details = subprocess.check_output([tools['llvm-objdump'], '-p', str(item)], text=True)
        imports = re.findall(r'DLL Name:\s*(\S+)', details)
        if not imports:
            raise RuntimeError('Could not inspect DLL dependencies: ' + str(item))
        for name in imports:
            lower = name.lower()
            if lower in system_dlls or lower.startswith(('api-ms-win-', 'ext-ms-win-')):
                continue
            if lower in {n.lower() for n in runtime}:
                continue
            dep = compiler_bin / name
            if not dep.is_file() or dep.is_symlink():
                raise RuntimeError('Missing runtime dependency: ' + str(dep))
            # This release statically links the compiler runtime. Reject new
            # dependencies until their distribution and licenses are reviewed.
            if lower != 'libpcre2-8-0.dll':
                raise RuntimeError('Unexpected runtime DLL: ' + name)
            runtime[name] = dep
            queue.append(dep)
    if 'libpcre2-8-0.dll' in {n.lower() for n in runtime}:
        licence = compiler_bin.parent / 'share/licenses/pcre2/LICENCE.md'
        if not licence.is_file():
            raise RuntimeError('PCRE2 license file missing: ' + str(licence))
        runtime['PCRE2-LICENCE.md'] = licence
    runtime['LICENSE'] = ROOT / 'LICENSE'

    stage = Path(tempfile.mkdtemp(prefix='.micius_cli-staging-', dir=ROOT))
    for name, path in runtime.items():
        shutil.copyfile(path, stage / name)
    clean_env = dict(os.environ, PATH=str(Path(os.environ['SystemRoot'])/'System32'))
    run('standalone-version', [stage/exe.name, '--version'], cwd=stage, env=clean_env)
    receipt = {'release': '0.7.3', 'preset': 'int8-release-windows',
               'source_manifest_sha256': hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
               'files': {name: hashlib.sha256((stage/name).read_bytes()).hexdigest() for name in sorted(runtime)}}
    (stage/'BUILD-RECEIPT.json').write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
    # Publish only after build, tests, source and dependency checks succeeded.
    # On failure, retain the staging/build data for diagnosis; never remove user files.
    stage.rename(target)
    print('\nREADY: ' + str(target/exe.name))
    print('From the source root: ./micius_cli/micius_cli.exe')
    print('Models and conversations are not moved. Build logs: ' + str(logs))


if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        print('BUILD STOPPED: ' + type(exc).__name__ + ': ' + str(exc), file=sys.stderr)
        sys.exit(1)
