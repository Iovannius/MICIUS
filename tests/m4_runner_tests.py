"""Exercise M4 delivery, full-logits checks and fail-closed evidence handling."""
import json
import copy
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile
from unittest.mock import patch
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import m4_run as run
import m4_selected as selected
from chat_text_tests import make_fixture, metadata, controlled
from reference.dense_fixture import fixture, write_gguf
from dense_text_tests import metadata as dense_metadata
from m3_evidence import core


def reject(fn):
    try: fn()
    except (ValueError, FileNotFoundError): return
    raise AssertionError('Invalid record accepted')


def main():
    session, probe, token, prompt = map(lambda p: Path(p).resolve(), sys.argv[1:])
    if os.name == 'nt':
        assert selected.text_file_arg(SimpleNamespace(resolve=lambda:'C:/first/second')) == '\\\\?\\C:\\first\\second'
        assert selected.text_file_arg(SimpleNamespace(resolve=lambda:'//server/share/file')) == '\\\\?\\UNC\\server\\share\\file'
    with tempfile.TemporaryDirectory(dir=Path.cwd(), prefix='m4-flow-') as tmp:
        root = Path(tmp);results = root / 'results';results.mkdir()
        models = root / 'models';models.mkdir()
        templates = json.loads((Path(__file__).parent / 'text/templates.json').read_bytes())
        protocol = {'context':256, 'messages':['first','next'], 'dense_tokens':list(range(1,9)), 'models':{},
                    'sampling':{'greedy':{'version':1,'seed':123,'max_new_tokens':4,'temperature':0},
                                'seeded':{'version':1,'seed':123,'max_new_tokens':4,'temperature':0.7,'top_k':12}}}
        golden = {}
        for name, arch, pre in [('smol','llama','smollm'),('qwen','qwen2','qwen2')]:
            model = models / (name+'.gguf')
            make_fixture(model, context=384, vocab=288, architecture=arch,
                         tokenizer_metadata=metadata(pre,templates[pre]), tensor_override=controlled)
            protocol['models'][name] = {'filename':model.name,'bytes':model.stat().st_size,'model_sha256':run.digest(model)}
            _,_,records = run.previous.step(session,model,root/'golden-cache',256,'-','first\nnext\n',root/('golden-'+name),
                                            'version=1\nseed=123\nmax_new_tokens=4\n')
            for backend in run.PROFILES:
                golden[backend+'-'+name] = {'turns':[{'finish':{0:'eot',1:'reply-limit'}[r['finish']],
                    'generated_tokens':r['tokens'],'context':r['context'],'utf8_sha256':r['utf8_sha256']} for r in records]}
        (root/'docs/evidence').mkdir(parents=True)
        run.save(root/'docs/evidence/m2-functional-result.json',{'checks':golden})
        run.save(root/'protocol.json',protocol);run.save(root/'manifest.json',{})
        dense = models/'dense.gguf'
        meta,tensors,_,_=fixture(dense,vocab=288,context=384)
        template=json.loads((Path(__file__).parent/'text/dense-templates.json').read_bytes())['Qwen-Qwen3-0.6B-slice-filters']
        meta.update(dense_metadata('qwen2',template))
        tensors['output.weight']=([256,288],0,bytes(256*288*4))
        write_gguf(dense,meta,tensors)
        def prepared(i):
            work=results/('m4-20000101-000000-'+f'{i:08x}');work.mkdir();(work/'logs').mkdir()
            shutil.copyfile(session,work/'session.exe');shutil.copyfile(probe,work/'probe.exe')
            shutil.copyfile(token,work/'token.exe');shutil.copyfile(prompt,work/'prompt.exe')
            tools={k:{'path':p,'sha256':run.digest(work/p)} for k,p in [('model','probe.exe'),('session','session.exe'),('cli','session.exe'),
                ('token','token.exe'),('prompt','prompt.exe')]}
            run.save(work/'prepared.json',{'unit':'M4','status':'PREPARED','dense_external_linkage':'PASS',
                'manifest_sha256':run.digest(root/'manifest.json'),'profiles':{b:{'m4_tools':tools} for b in run.PROFILES}})
            run.save(work/'status.json',{'status':'PREPARED','prepared_sha256':run.digest(work/'prepared.json')})
            return work
        with patch.multiple(run, ROOT=root, RESULTS=results, PROTOCOL=root/'protocol.json', MANIFEST=root/'manifest.json',verify=lambda:None):
            spec={'unit':'SYNTHETIC_SELECTED','version':1,'required_chat':True,'models':[{'filename':dense.name,
                'bytes':dense.stat().st_size,'sha256':run.digest(dense),'architecture':'qwen3',
                'weight_path':'gguf_block_dequant_ordered_f64','vocab':288}]}
            assert selected.pinned_paths(models,spec)==[dense.resolve()]
            invalid=copy.deepcopy(spec);invalid['models'][0]['sha256']='0'*64
            reject(lambda:selected.pinned_paths(models,invalid))
            invalid=copy.deepcopy(spec);invalid['models']*=2
            reject(lambda:selected.pinned_paths(models,invalid))
            good=prepared(1);proof=selected.preflight(good,[dense],spec)
            run.save(good/'selected-preflight.json',proof);run.selected_proof(good,proof)
            assert not (good/'LAUNCHED').exists() and not (good/'result.json').exists()
            reject(lambda:run.qualify(good,models,[dense],chat_dense=False,selected=proof))
            assert not (good/'LAUNCHED').exists()
            record=good/'observations/selected-preflight/checks.json';before=record.read_bytes()
            run.save(record,{'status':'PASS','checks':[]})
            invalid=copy.deepcopy(proof);invalid['evidence_sha256'][record.relative_to(good).as_posix()]=run.digest(record)
            reject(lambda:run.selected_proof(good,invalid));record.write_bytes(before)
            run.qualify(good,models,[dense],chat_dense=True,selected=proof)
            result=run.read(good/'result.json')
            assert result['status']=='M4_SELECTED_MODELS_FUNCTIONAL_PASS' and len(result['checks'])==8 and len(result['additional'])==2
            assert result['selected_experiment']=='SYNTHETIC_SELECTED'
            assert all(not x['independent_real_numerical_reference'] for x in result['additional'].values())
            assert all(c['status']=='PASS' for x in result['additional'].values() for c in x['chat'].values())
            raw=(good/'observations/INT8-additional-1-seeded/whole/state.bin').read_bytes()
            decoded=core(raw);dims=decoded['dimensions']
            assert decoded['feature_flags']==7 and dims[0]!=dims[4]*dims[6]
            # Unknown features and independent widths without the file-weight
            # flag must remain rejected by the independent evidence decoder.
            for flags in (0,1,5,16):
                invalid=bytearray(raw);struct.pack_into('<I',invalid,124,flags)
                reject(lambda:core(invalid))
            reject(lambda:run.qualify(good,models,[dense]))
            stopped=prepared(2)
            def interruption(folder,*a,**kw):
                folder.mkdir(parents=True);(folder/'partial.txt').write_text('retained');raise KeyboardInterrupt()
            with patch.object(run,'scenario',interruption):
                try:run.qualify(stopped,models)
                except KeyboardInterrupt:pass
                else:raise AssertionError('Interruption accepted')
            assert run.read(stopped/'status.json')['status']=='STOPPED' and not (stopped/'result.json').exists()
            reject(lambda:run.qualify(stopped,models))
            changed=prepared(3);(changed/'probe.exe').write_bytes(b'bad');reject(lambda:run.qualify(changed,models))
            assert not (changed/'LAUNCHED').exists()
            altered=good/'observations/selected-preflight/INT8-1-first/tokens.u32'
            before=altered.read_bytes();altered.write_bytes(b'bad')
            reject(lambda:run.selected_proof(good,proof));altered.write_bytes(before)
            record=good/'observations/INT8-additional-1/0/logits.f32';record.write_bytes(b'bad')
            reject(lambda:run.archive(good,True))
            # No-new-model pass is explicitly limited; do not claim dense acceptance.
            pending=prepared(4)
            with patch.object(run,'scenario',lambda *a,**kw:{'model_sha256':protocol['models']['smol' if 'smol' in str(a[0]) else 'qwen']['model_sha256']}):
                (pending/'observations').mkdir();run.qualify(pending,models)
            assert run.read(pending/'result.json')['status']=='M4_Q8_REGRESSION_PASS_NEW_MODELS_PENDING'
        inputs=root/'ids';inputs.write_bytes(struct.pack('<8I',*range(1,9)))
        # Nested source checkouts can place evidence past Windows MAX_PATH.
        # Exercise both CreateProcess's cwd limit and the probe's file writes.
        deep=root/('nested-'+'a'*100)/('evidence-'+'b'*100)
        assert len(str(deep))>260
        program=root/'programs'/('bin-'+'c'*100)/('tools-'+'d'*100)/'probe.exe'
        program.parent.mkdir(parents=True);shutil.copyfile(probe,program)
        assert len(str(program))>260
        run.forward(deep,program,dense,inputs)
        original=run.native
        def truncated(args,folder,expected=0):
            original(args,folder,expected);(folder/'logits.f32').write_bytes(b'')
        with patch.object(run,'native',truncated):
            reject(lambda:run.forward(root/'truncated',probe,dense,inputs))
    print('M4 synthetic orchestration, replay, missing/changed evidence and interruption: PASS')


if __name__=='__main__': main()
