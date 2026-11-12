"""Exercise the delivered continuation checker and evidence rejection, synthetic only."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
import zipfile
from chat_text_tests import make_fixture, metadata, controlled

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import m3_run as run
from m3_evidence import snapshot


def reject(fn):
    try:
        fn()
    except (ValueError, FileNotFoundError):
        return
    raise AssertionError('Invalid evidence accepted')


def orchestration(root, probe, templates):
    """Run the actual delivery loop on tiny fixtures, never target models.

    Both labels deliberately use this test's backend binary. Each CTest build
    runs this independently; this is flow coverage, not a cross-backend claim.
    """
    root.mkdir()
    results = root / 'results';results.mkdir()
    models = root / 'models';models.mkdir()
    protocol = {'context': 256, 'messages': ['first', 'next'], 'models': {},
                'sampling': {'greedy': {'version': 1, 'seed': 123, 'max_new_tokens': 8, 'temperature': 0},
                             'seeded': {'version': 1, 'seed': 123, 'max_new_tokens': 8,
                                        'temperature': 0.8, 'top_k': 12, 'top_p': 0.9}}}
    golden = {}
    for name, arch, profile in [('smol', 'llama', 'smollm'), ('qwen', 'qwen2', 'qwen2')]:
        model = models / (name + '.gguf')
        make_fixture(model, context=384, vocab=288, architecture=arch,
                     tokenizer_metadata=metadata(profile, templates[profile]), tensor_override=controlled)
        protocol['models'][name] = {'filename': model.name, 'bytes': model.stat().st_size,
                                     'model_sha256': run.digest(model)}
        _, _, records = run.step(probe, model, root/'golden-cache', 256, '-', 'first\nnext\n',
                                  root/('golden-'+name), 'version=1\nmax_new_tokens=8\nseed=123\n')
        for backend in ('INT8', 'F64'):
            golden[backend+'-'+name] = {'turns': [{'finish': {0:'eot',1:'reply-limit'}[r['finish']],
                'generated_tokens': r['tokens'], 'context': r['context'], 'utf8_sha256': r['utf8_sha256']} for r in records]}
    (root/'docs/evidence').mkdir(parents=True)
    run.save(root/'docs/evidence/m2-functional-result.json', {'checks': golden})
    run.save(root/'protocol.json', protocol)
    run.save(root/'manifest.json', {'synthetic_flow_only': True})
    def prepared(index):
        work = results / ('m3-20000101-000000-'+f'{index:08x}');work.mkdir()
        (work/'logs').mkdir()
        import shutil
        shutil.copyfile(probe, work/'probe.exe')
        receipt = {'unit':'M3','status':'PREPARED','manifest_sha256':run.digest(root/'manifest.json'),
                   'profiles': {b:{'session_tools':{k:{'path':'probe.exe','sha256':run.digest(work/'probe.exe')}
                                for k in ('session','cli')}} for b in ('INT8','F64')}}
        run.save(work/'prepared.json', receipt)
        run.save(work/'status.json', {'status':'PREPARED','prepared_sha256':run.digest(work/'prepared.json')})
        return work
    with patch.multiple(run, ROOT=root, RESULTS=results, PROTOCOL=root/'protocol.json',
                        MANIFEST=root/'manifest.json', verify=lambda: None):
        good=prepared(1);run.qualify(good,models)
        result=run.read(good/'result.json')
        assert run.read(good/'status.json')['status']=='PERSISTENT_CHAT_FUNCTIONAL_PASS' and len(result['checks'])==8
        for flavor in ('REVIEW','FULL'):
            assert (root/'exports'/f'{good.name}-{flavor}.zip').is_file()
        reject(lambda: run.qualify(good,models))
        stopped=prepared(2)
        def interrupted(folder,*args,**kwargs):
            folder.mkdir(parents=True);(folder/'partial.txt').write_text('retained')
            raise KeyboardInterrupt()
        with patch.object(run,'scenario',interrupted):
            try:run.qualify(stopped,models)
            except KeyboardInterrupt:pass
            else:raise AssertionError('Interruption accepted')
        assert run.read(stopped/'status.json')['status']=='STOPPED'
        assert list((stopped/'observations').rglob('partial.txt')) and not (stopped/'result.json').exists()
        reject(lambda: run.qualify(stopped,models))
        changed=prepared(3);(changed/'probe.exe').write_bytes(b'changed')
        reject(lambda: run.qualify(changed,models))
        assert not (changed/'LAUNCHED').exists()
    original=run.native
    def missing(args, folder, expected=0):
        original(args,folder,expected)
        (folder/'events.txt').write_text('')
    with patch.object(run,'native',missing):
        reject(lambda:run.step(probe,models/'smol.gguf',root/'missing-cache',256,'-','first\n',root/'missing',
                               'version=1\nmax_new_tokens=8\n'))


def main():
    probe = Path(sys.argv[1]).resolve()
    templates = json.loads((Path(__file__).parent / 'text/templates.json').read_bytes())
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as temp:
        root = Path(temp)
        model = root / 'tiny.gguf'
        make_fixture(model, context=384, vocab=288, architecture='qwen2',
                     tokenizer_metadata=metadata('qwen2', templates['qwen2']), tensor_override=controlled)
        protocol = {'context': 256, 'messages': ['first', 'next']}
        conf = 'version=1\nseed=12345\nmax_new_tokens=8\ntemperature=0.8\ntop_k=12\ntop_p=0.9\nrepeat_penalty=1.1\nrepeat_last_n=64\n'
        answer = run.scenario(root / 'observations/test', probe, model, root / 'cache', protocol, conf)
        assert answer['status'] == 'PASS'
        sample = root / 'observations/test/resume/snapshot.mcs'
        raw = sample.read_bytes()
        reject(lambda: snapshot(raw[:-1]))
        wrong = bytearray(raw);wrong[0] ^= 1;wrong[-64:] = hashlib.sha256(wrong[:-64]).hexdigest().encode()
        reject(lambda: snapshot(wrong))
        (root / 'prepared.json').write_text('{}')
        (root / 'status.json').write_text('{}')
        run.save(root / 'result.json', {'evidence_sha256': {'observations/test/resume/snapshot.mcs': run.digest(sample),
                                                          'prepared.json': run.digest(root / 'prepared.json')}})
        # An archived result must never silently adopt changed live evidence.
        sample.write_bytes(raw[:-1])
        run.ROOT = root
        reject(lambda: run.archive(root, False))
        sample.write_bytes(raw)
        run.archive(root, False); run.archive(root, True)
        with zipfile.ZipFile(root / 'exports' / (root.name + '-REVIEW.zip')) as z:
            assert 'observations/test/resume/snapshot.mcs' not in z.namelist()
        with zipfile.ZipFile(root / 'exports' / (root.name + '-FULL.zip')) as z:
            assert z.read('observations/test/resume/snapshot.mcs') == raw
        sample.unlink();reject(lambda: run.archive(root, False))
        # Check a rejected native process cannot satisfy a successful step.
        reject(lambda: run.step(probe, model, root/'cache', 256, 'bad-id', 'x\n', root/'bad-step'))
        orchestration(root/'full-flow',probe,templates)
    print('M3 runner continuation, bounded parser, missing/changed evidence rejection and archives: PASS')


if __name__ == '__main__':
    main()
