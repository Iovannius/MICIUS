"""M3 preparation and operator-run functional checks. No performance sampling."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import uuid
import zipfile

from m1_prepare import ROOT, PROFILES, digest, execute, prepare, require, save
from m3_evidence import snapshot, same_reset
from verify_release import MANIFEST, verify

RESULTS = ROOT / 'results'
PROTOCOL = ROOT / 'validation/m3/protocol.json'


def read(path):
    return json.loads(path.read_bytes())


def native(args, folder, expected=0):
    # CreateProcessW cannot use an over-MAX_PATH current directory, even with
    # the extended path prefix. Evidence/model paths remain absolute; only the
    # process working directory may use a shorter existing ancestor.
    args = [str(arg.resolve()) if isinstance(arg, Path) else str(arg) for arg in args]
    cwd = folder.resolve()
    executable = None
    if os.name == 'nt':
        while len(str(cwd)) >= 248:
            require(cwd.parent != cwd, 'No usable process working directory')
            cwd = cwd.parent
        # With lpApplicationName=NULL, Windows limits the program-name portion
        # of the command line to MAX_PATH. Pass its extended absolute path
        # separately when the copied verification executable itself is deep.
        if len(args[0]) >= 248:
            executable = str(Path(args[0]).resolve()).replace('/', '\\')
            if not executable.startswith('\\\\?\\'):
                executable = ('\\\\?\\UNC\\' + executable[2:] if executable.startswith('\\\\')
                              else '\\\\?\\' + executable)
    save(folder / 'command.json', {'argv': args, 'cwd': str(cwd), 'executable': executable})
    started = time.monotonic()
    print('  ' + folder.name, flush=True)
    with (folder / 'stdout.txt').open('wb') as out, (folder / 'stderr.txt').open('wb') as err:
        child = subprocess.Popen(args, executable=executable, cwd=cwd, stdout=out, stderr=err)
        try:
            while True:
                try:
                    code = child.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    print(f'  {folder.name} 正在运行，已等待 {time.monotonic()-started:.0f} 秒', flush=True)
        except BaseException:
            if child.poll() is None:
                child.kill()
            child.wait()
            save(folder / 'exit.json', {'returncode': child.returncode, 'interrupted': True})
            raise
    save(folder / 'exit.json', {'returncode': code, 'interrupted': False})
    tail = (folder / 'stderr.txt').read_bytes()[-2000:].decode('utf-8', errors='replace') if code != expected else ''
    require(code == expected, f'Native exit {code}; see {folder}/stderr.txt\n{tail}')


def step(probe, model, cache, capacity, session, text, out, config=None, expected=0):
    out.mkdir(parents=True)
    (out / 'stdin.txt').write_bytes(text.encode('utf-8'))
    args = [probe, model, cache, capacity, session, out / 'stdin.txt', out]
    if config is not None:
        (out / 'new-config.conf').write_bytes(config.encode('utf-8'))
        args.append(out / 'new-config.conf')
    native(args, out, expected)
    if expected:
        return None, None, []
    require(b'SESSION_CHECK_PASS' in (out / 'stdout.txt').read_bytes(), 'Missing native completion')
    directory = Path((out / 'directory.txt').read_text(encoding='utf-8')).resolve()
    require(cache.resolve() in directory.parents and re.fullmatch('[0-9a-f]{32}', directory.name), 'Bad context location')
    shutil.copyfile(directory / 'kv/chat.mcs', out / 'snapshot.mcs')
    shutil.copyfile(directory / 'sampling.conf', out / 'sampling.conf')
    evidence, raw = snapshot((out / 'snapshot.mcs').read_bytes())
    require(raw == (out / 'state.bin').read_bytes(), 'Exported engine and snapshot differ')
    records = []
    for i, line in enumerate((out / 'events.txt').read_text().splitlines(), 1):
        numbers = list(map(int, line.split()))
        require(len(numbers) == 4, 'Bad event record')
        turn, finish, tokens, context = numbers
        require(turn == i and finish in (0, 1) and 0 < tokens <= int(evidence['config']['max_new_tokens'])
                and 0 < context <= capacity, 'Incomplete or context-limited turn')
        answer = (out / f'turn-{i}.utf8').read_bytes()
        # Raw byte-BPE output may end in a partial UTF-8 sequence at the token
        # budget. Equality checks retain these bytes without normalizing them.
        records.append({'finish': finish, 'tokens': tokens, 'context': context,
                        'utf8_sha256': hashlib.sha256(answer).hexdigest()})
    require(len(records) == len([x for x in text.splitlines() if x and not x.startswith(':')]), 'Missing turn records')
    if records:
        require(records[-1]['context'] == evidence['core']['sequence'], 'Final event/KV mismatch')
    save(out / 'decoded.json', evidence)
    return directory, evidence, records


def scenario(folder, probe, model, cache, protocol, config, golden=None):
    folder.mkdir(parents=True)
    first, second = protocol['messages']
    cap = protocol['context']
    a, whole, total = step(probe, model, cache, cap, '-', first + '\n' + second + '\n', folder / 'whole', config)
    b, initial, one = step(probe, model, cache, cap, '-', first + '\n', folder / 'first', config)
    _, restored, two = step(probe, model, cache, cap, b.name, second + '\n', folder / 'resume')
    require(a != b and a.parent == b.parent, 'Conversations not independently isolated')
    require(total == one + two, 'Split-process output/events differ')
    require((folder / 'whole/state.bin').read_bytes() == (folder / 'resume/state.bin').read_bytes(), 'KV/logits continuation differs')
    require((folder / 'whole/snapshot.mcs').read_bytes() == (folder / 'resume/snapshot.mcs').read_bytes(), 'History/config/RNG continuation differs')
    for source, target in [('whole/turn-1.utf8', 'first/turn-1.utf8'), ('whole/turn-2.utf8', 'resume/turn-1.utf8')]:
        require((folder / source).read_bytes() == (folder / target).read_bytes(), 'Reply bytes differ')
    require(whole['turns'] == 2 and initial['turns'] == 1 and total[1]['context'] > total[0]['context'], 'Turn/history not advanced')
    require(whole['config'] == initial['config'], 'Effective configuration differs')
    values = dict(line.split('=', 1) for line in config.splitlines() if line)
    for key in values:
        require(float(values[key]) == float(whole['config'][key]) if key in ('temperature','top_p','repeat_penalty')
                else int(values[key]) == int(whole['config'][key]), 'Ignored sampling configuration')
    require(whole['rng'] == int(values['seed']) if float(values['temperature']) == 0
            else whole['rng'] != int(values['seed']), 'Unexpected random state transition')
    before_a = digest(a / 'kv/chat.mcs')
    _, _, replay = step(probe, model, cache, cap, b.name, ':reset\n' + first + '\n', folder / 'reset')
    require(replay == one, 'Reset reply/event differs')
    require((folder / 'reset/turn-1.utf8').read_bytes() == (folder / 'first/turn-1.utf8').read_bytes(), 'Reset reply bytes differ')
    same_reset((folder / 'reset/snapshot.mcs').read_bytes(), (folder / 'first/snapshot.mcs').read_bytes())
    require(digest(a / 'kv/chat.mcs') == before_a, 'Reset affected another conversation')
    if golden is not None:
        for got, expected in zip(total, golden):
            require(got == {'finish': {'eot': 0, 'reply-limit': 1}[expected['finish']],
                            'tokens': expected['generated_tokens'], 'context': expected['context'],
                            'utf8_sha256': expected['utf8_sha256']}, 'Greedy output differs from accepted M2')
        require(len(total) == len(golden), 'Incomplete historical comparison')
    return {'status': 'PASS', 'turns': total, 'saved_rng': whole['rng'],
            'history_tokens': len(whole['tokens']), 'snapshot_bytes': (folder / 'whole/snapshot.mcs').stat().st_size,
            'active_kv_bytes': whole['core']['active_kv_bytes'], 'model_sha256': whole['model_sha256'],
            'numerical_contract': whole['core']['contract'], 'greedy_M2_match': golden is not None}


def build(unit='M3'):
    RESULTS.mkdir(exist_ok=True)
    name = unit.lower() + '-' + datetime.datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8]
    work = RESULTS / name
    work.mkdir()
    print('WORK', work, flush=True)
    save(work / 'status.json', {'status': 'PREPARING'})
    try:
        receipt = prepare(work)
        receipt['unit'] = unit
        for backend in PROFILES:
            prefix = work / 'sdk' / backend
            for module in ('session', 'sampling'):
                external = work / f'{module}-consumer-{backend}'
                execute(['cmake', '-S', ROOT / f'examples/{module}-consumer', '-B', external, '-G', 'Ninja',
                         '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_C_COMPILER=clang', '-DCMAKE_CXX_COMPILER=clang++',
                         '-DMICIUS_DIR=' + str(prefix / 'lib/cmake/MICIUS')], work, work / 'logs' / f'{backend}-{module}-configure.log')
                link = execute(['cmake', '--build', external, '--verbose'], work, work / 'logs' / f'{backend}-{module}-link.log')
                if module == 'sampling':
                    require('libmicius_core.a' not in link and 'libmicius_text.a' not in link and 'pcre2' not in link.lower(),
                            'Sampler acquired an inference/text dependency')
                    execute([external / 'micius_sampling_probe.exe'], work, work / 'logs' / f'{backend}-sampling-run.log')
            tools = {'cli': prefix / 'bin/micius_cli.exe',
                     'session': work / f'session-consumer-{backend}/micius_session_probe.exe'}
            receipt['profiles'][backend]['session_tools'] = {k: {'path': p.relative_to(work).as_posix(), 'sha256': digest(p)} for k, p in tools.items()}
        # Installed C interfaces must also reject cross-backend snapshots.
        sys.path.insert(0, str(ROOT / 'tests'))
        from chat_text_tests import make_fixture, metadata, controlled
        templates = read(ROOT / 'tests/text/templates.json')
        model = work / 'tiny-chat.gguf'
        make_fixture(model, context=384, vocab=288, architecture='llama', tokenizer_metadata=metadata('smollm', templates['smollm']), tensor_override=controlled)
        p = lambda b: work / receipt['profiles'][b]['session_tools']['session']['path']
        cache = work / 'synthetic-cache'
        config = 'version=1\nseed=7\nmax_new_tokens=8\n'
        d, _, _ = step(p('INT8'), model, cache, 256, '-', 'test\n', work / 'installed-chat', config)
        old = digest(d / 'kv/chat.mcs')
        step(p('F64'), model, cache, 256, d.name, 'next\n', work / 'backend-reject', expected=1)
        require(digest(d / 'kv/chat.mcs') == old, 'Rejected cross-backend import changed saved state')
        require('contract/shape/FP' in (work / 'backend-reject/stderr.txt').read_text(), 'Unexpected cross-backend failure')
        receipt['cross_backend_snapshot_rejected'] = True
        save(work / 'prepared.json', receipt)
        identity = digest(work / 'prepared.json')
        save(work / 'status.json', {'status': 'PREPARED', 'prepared_sha256': identity})
        save(RESULTS / (unit+'-last-prepared.json'), {'name': name, 'prepared_sha256': identity})
    except BaseException as e:
        save(work / 'status.json', {'status': 'PREPARATION_STOPPED', 'error': str(e), 'type': type(e).__name__})
        raise
    print('PREPARED', work, flush=True)
    return work


def validate(work):
    require(work.resolve().parent == RESULTS.resolve() and re.fullmatch(r'm3-\d{8}-\d{6}-[0-9a-f]{8}', work.name), 'Expected an M3 result directory')
    verify()
    receipt, state = read(work / 'prepared.json'), read(work / 'status.json')
    require(receipt['unit'] == 'M3' and receipt['status'] == 'PREPARED' and receipt['manifest_sha256'] == digest(MANIFEST)
            and digest(work / 'prepared.json') == state['prepared_sha256'], 'Changed source/preparation')
    bins = {}
    for b in PROFILES:
        bins[b] = {}
        for k in ('cli', 'session'):
            tool = receipt['profiles'][b]['session_tools'][k]
            p = (work / tool['path']).resolve()
            require(work.resolve() in p.parents and p.is_file() and digest(p) == tool['sha256'], 'Changed or missing executable')
            bins[b][k] = p
    return bins


def archive(work, full):
    flavor = 'FULL' if full else 'REVIEW'
    target = ROOT / 'exports' / f'{work.name}-{flavor}.zip'
    target.parent.mkdir(exist_ok=True)
    result = read(work / 'result.json')
    for name, expected in result['evidence_sha256'].items():
        p = (work / name).resolve()
        require(work.resolve() in p.parents and p.is_file() and digest(p) == expected, 'Evidence changed before archive')
    paths = [work / p for p in result['evidence_sha256'] if full or Path(p).suffix not in ('.mcs', '.bin')]
    paths += [work / 'result.json', work / 'status.json']
    part = target.with_suffix('.part')
    with zipfile.ZipFile(part, 'x', compression=zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p, p.relative_to(work).as_posix())
    with zipfile.ZipFile(part) as z:
        require(z.testzip() is None and len(z.namelist()) == len(paths), 'Incomplete archive')
        for p in paths:
            with z.open(p.relative_to(work).as_posix()) as src:
                h = hashlib.sha256()
                for block in iter(lambda: src.read(4 * 1024 * 1024), b''):
                    h.update(block)
            require(h.hexdigest() == digest(p), 'Archive differs from evidence')
    part.rename(target)
    print('ARCHIVED', target, flush=True)


def qualify(work, models):
    bins = validate(work)
    state = read(work / 'status.json')
    require(state['status'] == 'PREPARED', 'Expected a fresh prepared run; no stitching after interruption')
    with (work / 'LAUNCHED').open('x', encoding='utf-8') as f:
        f.write(str(uuid.uuid4()) + '\n')
    try:
        save(work / 'status.json', {**state, 'status': 'RUNNING'})
        protocol = read(PROTOCOL)
        golden = read(ROOT / 'docs/evidence/m2-functional-result.json')['checks']
        paths = {}
        for name, info in protocol['models'].items():
            p = models / info['filename']
            print('核对实机模型身份：', info['filename'], flush=True)
            require(p.is_file() and p.stat().st_size == info['bytes'] and digest(p) == info['model_sha256'], 'Model identity mismatch')
            paths[name] = p
        checks = {}
        for backend in PROFILES:
            for name, model in paths.items():
                for mode, conf in protocol['sampling'].items():
                    label = f'{backend}-{name}-{mode}'
                    print(f'SESSION {len(checks)+1}/8 {label}：两轮、跨进程续聊、reset 重放', flush=True)
                    config = ''.join(f'{k}={v}\n' for k, v in conf.items())
                    checks[label] = scenario(work / 'observations' / label, bins[backend]['session'], model,
                                             work / 'session-cache' / label, protocol, config,
                                             golden[f'{backend}-{name}']['turns'][:2] if mode == 'greedy' else None)
                    require(checks[label]['model_sha256'] == protocol['models'][name]['model_sha256'], 'Wrong snapshot model')
        require(len(checks) == 8, 'Missing scenario')
        # Recheck runnable identity and all supplied files after inference.
        validate(work)
        for name, p in paths.items():
            require(digest(p) == protocol['models'][name]['model_sha256'], 'Model changed during validation')
        files = [work / 'LAUNCHED', work / 'prepared.json']
        files += sorted((work / 'logs').glob('*.log'))
        files += sorted(p for p in (work / 'observations').rglob('*') if p.is_file())
        result = {'status': 'PERSISTENT_CHAT_FUNCTIONAL_PASS', 'scope': 'Same-build snapshot/config/RNG continuation; not a performance or broad model-quality claim',
                  'prepared_sha256': state['prepared_sha256'], 'protocol_sha256': digest(PROTOCOL), 'checks': checks,
                  'evidence_sha256': {p.relative_to(work).as_posix(): digest(p) for p in files}}
        save(work / 'result.json', result)
        save(work / 'status.json', {**state, 'status': result['status']})
        archive(work, False)
        archive(work, True)
        print(result['status'], work, flush=True)
    except BaseException as e:
        save(work / 'status.json', {**state, 'status': 'STOPPED', 'error': str(e), 'type': type(e).__name__})
        print('证据保留于', work, file=sys.stderr)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('all'); sub.add_parser('prepare')
    sub.add_parser('run').add_argument('directory', type=Path)
    c = sub.add_parser('chat')
    c.add_argument('model', choices=('smol', 'qwen')); c.add_argument('backend', nargs='?', default='INT8', choices=tuple(PROFILES))
    c.add_argument('--session'); c.add_argument('--context', type=int, default=1024)
    args = parser.parse_args()
    models = Path(os.environ.get('MICIUS_MODELS', str(Path.cwd() / 'models'))).resolve()
    if args.action == 'chat':
        pointer = RESULTS / 'M3-last-prepared.json'
        if pointer.is_file():
            ref = read(pointer); work = RESULTS / ref['name']
            require(work.resolve().parent == RESULTS.resolve() and digest(work / 'prepared.json') == ref['prepared_sha256'], 'Changed preparation pointer')
        else:
            work = build()
        bins = validate(work)
        info = read(PROTOCOL)['models'][args.model]
        model = models / info['filename']
        require(model.is_file() and digest(model) == info['model_sha256'], 'Model identity mismatch')
        command = [bins[args.backend]['cli'], '--chat', model, '--context', str(args.context)]
        if args.session:
            command += ['--session', args.session]
        with subprocess.Popen(list(map(str, command))) as child:
            while True:
                try:
                    return child.wait()
                except KeyboardInterrupt:
                    continue
    work = args.directory.resolve() if args.action == 'run' else build()
    if args.action != 'prepare':
        qualify(work, models)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit('M3 interrupted; evidence retained')
    except Exception as e:
        sys.exit(f'M3 stopped: {type(e).__name__}: {e}')
