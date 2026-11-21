"""Pinned Llama 3.2 / Qwen3 experiment. Preflight never binds or runs a model."""
import argparse
import os
from pathlib import Path
import struct
import sys

import m4_run as run

SELECTION = run.ROOT / 'validation/m4/selected-models.json'


def text_file_arg(path):
    """The small C text probes need explicit extended paths for CRT file I/O."""
    absolute = str(path.resolve())
    if os.name != 'nt':
        return absolute
    # MSYS2 Python may render native Windows paths with forward slashes.
    # Extended paths bypass Windows slash normalization, so normalize first.
    absolute = absolute.replace('/', '\\')
    if absolute.startswith('\\\\?\\'):
        return absolute
    return '\\\\?\\UNC\\' + absolute[2:] if absolute.startswith('\\\\') else '\\\\?\\' + absolute


def pinned_paths(models, selection):
    run.require(selection.get('version') == 1 and selection.get('required_chat') is True,
                'Unsupported selected-model protocol')
    protocol = run.read(run.PROTOCOL)
    entries = list(protocol['models'].values()) + selection['models']
    names = [item['filename'] for item in entries]
    run.require(len(set(names)) == len(names) and selection['models'], 'Duplicate or missing model selection')
    for item in entries:
        name = item['filename']
        run.require(Path(name).name == name and Path(name).suffix.lower() == '.gguf', 'Expected a GGUF filename')
        path = (models / name).resolve()
        run.require(path.parent == models.resolve() and path.is_file(), 'Missing selected GGUF: ' + name)
        expected = item.get('sha256', item.get('model_sha256'))
        print('核对模型身份：' + name, flush=True)
        run.require(path.stat().st_size == item['bytes'] and run.digest(path) == expected,
                    'Model identity differs: ' + name)
    return [(models / item['filename']).resolve() for item in selection['models']]


def preflight(work, paths, selection):
    """Only tokenizer and --prompt-only executables; no forward/session calls."""
    bins = run.validate(work)
    run.require(run.read(work / 'status.json')['status'] == 'PREPARED' and not (work / 'LAUNCHED').exists(),
                'Fresh preparation required')
    run.require(paths and len(paths) == len(selection['models']), 'Incomplete selected model list')
    base = work / 'observations/selected-preflight'
    run.require(not base.exists(), 'Preflight already exists; use all for a fresh experiment')
    base.mkdir(parents=True)
    run.save(base / 'selection.json', selection)
    protocol = run.read(run.PROTOCOL)
    checks, cross_backend = [], {}
    for backend in run.PROFILES:
        for index, (model, item) in enumerate(zip(paths, selection['models']), 1):
            for phase, message in (('ordinary', protocol['messages'][0]), ('first', protocol['messages'][0]),
                                   ('next', protocol['messages'][1])):
                folder = base / (backend + '-' + str(index) + '-' + phase)
                folder.mkdir()
                source, tokens = folder / 'input.utf8', folder / 'tokens.u32'
                source.write_bytes(message.encode('utf-8'))
                args = ([bins[backend]['token'], model, text_file_arg(source), text_file_arg(tokens)] if phase == 'ordinary' else
                        [bins[backend]['prompt'], '--prompt-only', model, text_file_arg(source), phase, text_file_arg(tokens)])
                run.native(args, folder)
                raw = tokens.read_bytes()
                run.require(raw and len(raw) % 4 == 0, 'Missing or malformed prompt tokens')
                ids = [row[0] for row in struct.iter_unpack('<I', raw)]
                run.require(len(ids) < protocol['context'] and all(0 <= token < item['vocab'] for token in ids),
                            'Prompt tokens exceed the declared vocabulary/context')
                if phase == 'ordinary':
                    run.require(b'roundtrip=PASS' in (folder / 'stdout.txt').read_bytes(), 'Tokenizer roundtrip failed')
                signature = run.digest(tokens)
                key = (index, phase)
                run.require(signature == cross_backend.setdefault(key, signature), 'Tokenizer differs between builds')
                checks.append({'backend': backend, 'model_sha256': item['sha256'], 'phase': phase,
                               'tokens': len(ids), 'tokens_sha256': signature})
    run.require(len(paths) == len(selection['models']) and len(checks) == len(paths) * len(run.PROFILES) * 3,
                'Incomplete metadata preflight')
    run.save(base / 'checks.json', {'status': 'PASS', 'scope': 'Metadata and text only; no inference', 'checks': checks})
    return {'status': 'METADATA_PREFLIGHT_PASS', 'selection': selection,
            'protocol_sha256': run.digest(run.PROTOCOL), 'prepared_sha256': run.digest(work / 'prepared.json'),
            'evidence_sha256': {p.relative_to(work).as_posix(): run.digest(p)
                               for p in sorted(base.rglob('*')) if p.is_file()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('all')
    check = sub.add_parser('preflight', help='Read-only model/tokenizer checks on a fresh prepared directory; no inference')
    check.add_argument('directory', type=Path)
    args = parser.parse_args()
    run.verify()
    selection = run.read(SELECTION)
    models = Path(os.environ.get('MICIUS_MODELS', str(Path.cwd() / 'models'))).resolve()
    paths = pinned_paths(models, selection)
    work = run.build() if args.action == 'all' else args.directory.resolve()
    try:
        proof = preflight(work, paths, selection)
        run.save(work / 'selected-preflight.json', proof)
        run.selected_proof(work, proof)
    except BaseException as error:
        state = run.read(work / 'status.json')
        # A metadata-only failure must not leave a misleading runnable state.
        # Keep completed/previously launched runs immutable on accidental reuse.
        if state['status'] == 'PREPARED' and not (work / 'LAUNCHED').exists():
            run.save(work / 'status.json', {**state, 'status': 'PREFLIGHT_STOPPED',
                                          'type': type(error).__name__, 'error': str(error)})
        print('预检证据保留于', work, file=sys.stderr)
        raise
    if args.action == 'preflight':
        print('METADATA_PREFLIGHT_PASS_NO_INFERENCE', work, flush=True)
        return 0
    print('固定实验：8 项 Q8 回归、4 项新模型前向、8 项新模型续聊；不作性能或质量判定。', flush=True)
    run.qualify(work, models, paths, chat_dense=True, selected=proof)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit('M4-P1 interrupted; evidence retained; rerun all into a fresh directory')
    except Exception as error:
        sys.exit('M4-P1 stopped: ' + type(error).__name__ + ': ' + str(error))
