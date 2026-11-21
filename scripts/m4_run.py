"""M4 dense-format functional qualification. Real models run only on operator request."""
import argparse
import hashlib
import math
import os
from pathlib import Path
import re
import shutil
import struct
import sys
import uuid

import m3_run as previous
from m1_prepare import ROOT, PROFILES, digest, execute, require, save
from verify_release import MANIFEST, verify

RESULTS = ROOT / 'results'
PROTOCOL = ROOT / 'validation/m4/protocol.json'
read = previous.read
native = previous.native
scenario = previous.scenario


def build():
    work = previous.build('M4')
    save(work / 'status.json', {'status': 'PREPARING_M4'})
    try:
        receipt = read(work / 'prepared.json')
        sys.path.insert(0, str(ROOT / 'tests'))
        from reference.dense_fixture import fixture
        model = work / 'tiny-qwen3-mixed.gguf'
        fixture(model)
        inputs = work / 'dense-ids.u32'
        inputs.write_bytes(struct.pack('<8I', *read(PROTOCOL)['dense_tokens']))
        for backend in PROFILES:
            prefix = work / 'sdk' / backend
            outputs = []
            for mode in ('qwen3', 'all'):
                external = work / f'dense-{backend}-{mode}'
                execute(['cmake', '-S', ROOT / 'examples/file-consumer', '-B', external, '-G', 'Ninja',
                         '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_C_COMPILER=clang', '-DCMAKE_CXX_COMPILER=clang++',
                         '-DMICIUS_DIR=' + str(prefix / 'lib/cmake/MICIUS'), '-DPARSERS=' + mode],
                        work, work / 'logs' / f'{backend}-{mode}-configure.log')
                link = execute(['cmake', '--build', external, '--verbose'], work, work / 'logs' / f'{backend}-{mode}-link.log')
                for arch in ('llama', 'qwen2', 'qwen3'):
                    require(('libmicius_parser_' + arch + '.a' in link) == (mode == 'all' or arch == 'qwen3'),
                            'Wrong parser linkage')
                probe = external / 'micius_model_probe.exe'
                record = forward(work / f'synthetic-{backend}-{mode}', probe, model, inputs)
                outputs.append(record['logits_sha256'])
                if mode == 'qwen3':
                    text = execute([probe, work / 'tiny-llama.gguf', work / 'unlinked-cache', inputs, work / 'unexpected.f32'],
                                   work, work / 'logs' / f'{backend}-unlinked-llama.log', 1)
                    require('No linked parser for architecture: llama' in text, 'Unlinked architecture not rejected')
            require(outputs[0] == outputs[1], 'Linkage changed output')
            session = work / f'dense-session-{backend}'
            execute(['cmake', '-S', ROOT / 'examples/session-consumer', '-B', session, '-G', 'Ninja',
                     '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_C_COMPILER=clang', '-DCMAKE_CXX_COMPILER=clang++',
                     '-DMICIUS_DIR=' + str(prefix / 'lib/cmake/MICIUS'), '-DMICIUS_CONSUMER_QWEN3=ON'],
                    work, work / 'logs' / f'{backend}-dense-session-configure.log')
            execute(['cmake', '--build', session], work, work / 'logs' / f'{backend}-dense-session-link.log')
            paths = {'model': work / f'dense-{backend}-all/micius_model_probe.exe',
                     'session': session / 'micius_session_probe.exe', 'cli': prefix / 'bin/micius_cli.exe',
                     'token': work / f'build-{backend}/bin/m2_token_probe.exe',
                     'prompt': work / f'build-{backend}/bin/m2_chat_contract.exe'}
            receipt['profiles'][backend]['m4_tools'] = {k: {'path': p.relative_to(work).as_posix(), 'sha256': digest(p)}
                                                       for k, p in paths.items()}
        receipt['dense_external_linkage'] = 'PASS'
        save(work / 'prepared.json', receipt)
        identity = digest(work / 'prepared.json')
        save(work / 'status.json', {'status': 'PREPARED', 'prepared_sha256': identity})
        save(RESULTS / 'M4-last-prepared.json', {'name': work.name, 'prepared_sha256': identity})
    except BaseException as e:
        save(work / 'status.json', {'status': 'PREPARATION_STOPPED', 'type': type(e).__name__, 'error': str(e)})
        raise
    print('M4 PREPARED', work, flush=True)
    return work


def validate(work):
    require(work.resolve().parent == RESULTS.resolve() and re.fullmatch(r'm4-\d{8}-\d{6}-[0-9a-f]{8}', work.name),
            'Expected an M4 result directory')
    verify()
    receipt, state = read(work / 'prepared.json'), read(work / 'status.json')
    require(receipt['unit'] == 'M4' and receipt['status'] == 'PREPARED' and receipt.get('dense_external_linkage') == 'PASS'
            and receipt['manifest_sha256'] == digest(MANIFEST) and digest(work / 'prepared.json') == state['prepared_sha256'],
            'Incomplete or changed preparation')
    bins = {}
    for b in PROFILES:
        bins[b] = {}
        for key in ('model', 'session', 'cli', 'token', 'prompt'):
            tool = receipt['profiles'][b]['m4_tools'][key]
            path = (work / tool['path']).resolve()
            require(work.resolve() in path.parents and path.is_file() and digest(path) == tool['sha256'], 'Changed executable')
            bins[b][key] = path
    return bins


def forward(folder, probe, model, inputs):
    """Finite full logits, reset replay in C and an independent process replay."""
    folder.mkdir(parents=True)
    rows = len(inputs.read_bytes()) // 4
    hashes = []
    info = None
    for i in range(2):
        out = folder / str(i); out.mkdir()
        native([probe, model, folder / 'cache', inputs, out / 'logits.f32'], out)
        text = (out / 'stdout.txt').read_text(encoding='utf-8')
        data = dict(line.split('=', 1) for line in text.splitlines() if '=' in line)
        require(data.get('steps') == str(rows) and data.get('reset_replay') == 'PASS', 'Incomplete forward/reset')
        require(data.get('architecture') in ('llama', 'qwen2', 'qwen3') and
                data.get('weight_path') in ('q8_32', 'gguf_block_dequant_ordered_f64'), 'Unknown execution route')
        vocab = int(data['vocab'])
        raw = (out / 'logits.f32').read_bytes()
        require(vocab > 0 and len(raw) == rows * vocab * 4, 'Missing logits')
        require(all(math.isfinite(x[0]) for x in struct.iter_unpack('<f', raw)), 'Nonfinite logits')
        hashes.append(hashlib.sha256(raw).hexdigest())
        values = {k: data[k] for k in ('architecture', 'weight_path', 'vocab')}
        require(info is None or info == values, 'Changed runtime identity')
        info = values
    require(hashes[0] == hashes[1], 'Process replay differs')
    return {**info, 'status': 'FUNCTIONAL_PASS', 'steps': rows, 'logits_sha256': hashes[0],
            'reset_replay': True, 'process_replay': True, 'independent_real_numerical_reference': False}


def archive(work, full):
    # Shared archive implementation validates every evidence hash and ZIP member.
    old = previous.ROOT
    try:
        previous.ROOT = ROOT
        previous.archive(work, full)
    finally:
        previous.ROOT = old


def selected_proof(work, proof):
    require(proof.get('status') == 'METADATA_PREFLIGHT_PASS' and proof['selection'].get('required_chat') is True,
            'Missing selected-model preflight')
    require(proof.get('prepared_sha256') == digest(work / 'prepared.json') and
            proof.get('protocol_sha256') == digest(PROTOCOL), 'Preflight preparation/protocol differs')
    for name, expected in proof['evidence_sha256'].items():
        path = (work / name).resolve()
        require(work.resolve() in path.parents and path.is_file() and digest(path) == expected,
                'Selected-model preflight evidence missing or changed')
    require(proof['evidence_sha256'], 'Empty selected-model evidence')
    base = work / 'observations/selected-preflight'
    files = {p.relative_to(work).as_posix() for p in base.rglob('*') if p.is_file()}
    require(files == set(proof['evidence_sha256']), 'Preflight file set differs')
    require(read(base / 'selection.json') == proof['selection'], 'Preflight selection differs')
    checks = read(base / 'checks.json')
    expected = {(backend, item['sha256'], phase) for backend in PROFILES
                for item in proof['selection']['models'] for phase in ('ordinary', 'first', 'next')}
    observed = [(c['backend'], c['model_sha256'], c['phase']) for c in checks['checks']]
    require(checks['status'] == 'PASS' and len(observed) == len(expected) and set(observed) == expected,
            'Incomplete selected-model preflight records')


def qualify(work, models, dense=(), chat_dense=False, selected=None):
    bins = validate(work)
    state = read(work / 'status.json')
    require(state['status'] == 'PREPARED', 'Fresh run required; interrupted evidence is never stitched')
    if selected is not None:
        selected_proof(work, selected)
        require(chat_dense, 'Selected-model protocol requires chat; forward-only cannot pass')
    with (work / 'LAUNCHED').open('x', encoding='utf-8') as f:
        f.write(str(uuid.uuid4()) + '\n')
    try:
        save(work / 'status.json', {**state, 'status': 'RUNNING'})
        protocol = read(PROTOCOL)
        golden = read(ROOT / 'docs/evidence/m2-functional-result.json')['checks']
        paths, identities = {}, {}
        for name, item in protocol['models'].items():
            path = (models / item['filename']).resolve()
            require(path.is_file() and path.stat().st_size == item['bytes'] and digest(path) == item['model_sha256'],
                    'Q8 reference model identity mismatch')
            paths[name] = path; identities[str(path)] = item['model_sha256']
        supplied = list(dict.fromkeys(Path(p).resolve() for p in dense))
        for path in supplied:
            require(path.is_file() and path.suffix.lower() == '.gguf', 'Missing additional GGUF')
            identities[str(path)] = digest(path)
        if selected is not None:
            expected = selected['selection']['models']
            require(len(supplied) == len(expected) and len(expected) > 0, 'Selected model count differs')
            for path, item in zip(supplied, expected):
                require(path.name == item['filename'] and path.stat().st_size == item['bytes'] and
                        identities[str(path)] == item['sha256'], 'Selected model identity differs')
        save(work / 'run-inputs.json', {'models': identities, 'additional': list(map(str, supplied)),
                                      'additional_chat_requested': chat_dense, 'protocol_sha256': digest(PROTOCOL),
                                      'selected_preflight': selected})
        checks = {}
        for backend in PROFILES:
            for name, model in paths.items():
                for mode, conf in protocol['sampling'].items():
                    label = f'{backend}-{name}-{mode}'
                    print(f'Q8 REGRESSION {len(checks)+1}/8 {label}', flush=True)
                    checks[label] = scenario(work / 'observations' / label, bins[backend]['session'], model,
                        work / 'session-cache' / label, protocol, ''.join(f'{k}={v}\n' for k,v in conf.items()),
                        golden[f'{backend}-{name}']['turns'][:2] if mode == 'greedy' else None)
                    require(checks[label]['model_sha256'] == identities[str(model)], 'Wrong snapshot model')
        require(len(checks) == 8, 'Missing Q8 regression')
        inputs = work / 'dense-ids.u32';inputs.write_bytes(struct.pack('<8I', *protocol['dense_tokens']))
        additional = {}
        for i, model in enumerate(supplied):
            for backend in PROFILES:
                label = f'{backend}-additional-{i+1}'
                print('DENSE FUNCTIONAL', label, model.name, flush=True)
                item = forward(work / 'observations' / label, bins[backend]['model'], model, inputs)
                if selected is not None:
                    pinned = selected['selection']['models'][i]
                    require(all(item[key] == str(pinned[key]) for key in ('architecture', 'weight_path', 'vocab')),
                            'Selected model architecture, vocabulary or numerical route differs')
                item['model_sha256'] = identities[str(model)]
                if chat_dense:
                    item['chat'] = {}
                    for mode, conf in protocol['sampling'].items():
                        chat = label + '-' + mode
                        item['chat'][mode] = scenario(work / 'observations' / chat, bins[backend]['session'], model,
                            work / 'session-cache' / chat, protocol, ''.join(f'{k}={v}\n' for k,v in conf.items()))
                        require(item['chat'][mode]['model_sha256'] == identities[str(model)], 'Wrong additional snapshot model')
                additional[label] = item
        require(len(additional) == 2 * len(supplied), 'Missing additional model result')
        validate(work)
        if selected is not None:
            selected_proof(work, selected)
        for path, expected in identities.items():
            require(digest(Path(path)) == expected, 'Model changed during run')
        files = [work / p for p in ('LAUNCHED', 'prepared.json', 'run-inputs.json', 'dense-ids.u32')]
        if selected is not None:
            files.append(work / 'selected-preflight.json')
            require(read(work / 'selected-preflight.json') == selected, 'Preflight receipt differs')
        files += sorted((work / 'logs').glob('*.log'))
        files += sorted(p for p in (work / 'observations').rglob('*') if p.is_file())
        status = 'M4_Q8_REGRESSION_PASS_NEW_MODELS_PENDING' if not supplied else 'M4_SELECTED_MODELS_FUNCTIONAL_PASS'
        result = {'status': status, 'scope': 'Functional replay only. New formats still require independent real-model numerical/quality evaluation; no performance claim.',
                  'checks': checks, 'additional': additional, 'selected_experiment': selected['selection']['unit'] if selected else None,
                  'prepared_sha256': state['prepared_sha256'],
                  'protocol_sha256': digest(PROTOCOL), 'evidence_sha256': {p.relative_to(work).as_posix(): digest(p) for p in files}}
        save(work / 'result.json', result);save(work / 'status.json', {**state, 'status': status})
        archive(work, False);archive(work, True)
        print(status, work, flush=True)
    except BaseException as e:
        save(work / 'status.json', {**state, 'status': 'STOPPED', 'type': type(e).__name__, 'error': str(e)})
        print('证据保留于', work, file=sys.stderr)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    sub.add_parser('prepare')
    for action in ('all', 'run'):
        s = sub.add_parser(action)
        if action == 'run': s.add_argument('directory', type=Path)
        s.add_argument('--dense-model', action='append', type=Path, default=[])
        s.add_argument('--chat-dense', action='store_true', help='Also require known-template persisted chat for every additional model')
    args = p.parse_args()
    work = args.directory.resolve() if args.action == 'run' else build()
    if args.action != 'prepare':
        require(not args.chat_dense or args.dense_model, '--chat-dense requires --dense-model')
        models = Path(os.environ.get('MICIUS_MODELS', str(Path.cwd() / 'models'))).resolve()
        qualify(work, models, args.dense_model, args.chat_dense)
    return 0


if __name__ == '__main__':
    try: sys.exit(main())
    except KeyboardInterrupt: sys.exit('M4 interrupted; evidence retained')
    except Exception as e: sys.exit(f'M4 stopped: {type(e).__name__}: {e}')
