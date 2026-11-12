"""Independent-process snapshot and CLI checks on synthetic fixtures only."""
import hashlib,json,os,re,subprocess,sys,tempfile
from pathlib import Path
from chat_text_tests import make_fixture,metadata,controlled

def numerical_state(data):
    # Reset preserves the V1 lifetime diagnostic counter. It is not KV/logits
    # or sampling state. Keep every other byte in this comparison.
    assert data[:8]==b'MCSKV001' and len(data)>=148
    return data[:144]+b'\0'*4+data[148:]

def main():
    probe,contract,cli=map(lambda x:Path(x).resolve(),sys.argv[1:])
    templates=json.loads((Path(__file__).parent/'text/templates.json').read_text())
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
        root=Path(tmp);counter=0
        def run(args,text=b'',rc=0):
            p=subprocess.run(list(map(str,args)),input=text,capture_output=True,cwd=root,timeout=60,
                             env=dict(os.environ,MICIUS_CACHE_ROOT=str(root/'cli-cache')))
            assert p.returncode==rc,(args,p.returncode,p.stderr,p.stdout[-1500:]);return p
        def step(model,cache,session,lines,rc=0,context=256):
            nonlocal counter
            counter+=1;out=root/str(counter);out.mkdir();inp=out/'input';inp.write_text(lines,encoding='utf-8')
            run([probe,model,cache,context,session,inp,out],rc=rc)
            if rc:return out,None
            directory=Path((out/'directory.txt').read_text(encoding='utf-8'))
            return out,directory
        for arch,profile in [('llama','smollm'),('qwen2','qwen2')]:
            model=root/(arch+'.gguf')
            make_fixture(model,context=384,vocab=288,architecture=arch,tokenizer_metadata=metadata(profile,templates[profile]),tensor_override=controlled)
            cache=root/('cache-'+arch);run([contract,model,cache])
            for stochastic in [False,True]:
                _,a=step(model,cache,'-','');_,b=step(model,cache,'-','')
                config='version=1\nmax_new_tokens=8\n'
                if stochastic:config+='temperature=1.2\ntop_k=12\ntop_p=0.93\nrepeat_penalty=1.15\nrepeat_last_n=64\nseed=12345\n'
                for d in [a,b]:(d/'sampling.conf').write_text(config)
                all_at_once,_=step(model,cache,a.name,'first\nnext\n')
                first,_=step(model,cache,b.name,'first\n')
                first_snapshot=(b/'kv/chat.mcs').read_bytes()
                second,_=step(model,cache,b.name,'next\n')
                assert (all_at_once/'turn-1.utf8').read_bytes()==(first/'turn-1.utf8').read_bytes()
                assert (all_at_once/'turn-2.utf8').read_bytes()==(second/'turn-1.utf8').read_bytes()
                assert (all_at_once/'state.bin').read_bytes()==(second/'state.bin').read_bytes()
                assert (a/'kv/chat.mcs').read_bytes()==(b/'kv/chat.mcs').read_bytes()
                # Input model path changes do not change model identity; context shape does.
                step(model,cache,b.name,'',rc=1,context=128)
                reset,_=step(model,cache,b.name,':reset\nfirst\n')
                assert (reset/'turn-1.utf8').read_bytes()==(first/'turn-1.utf8').read_bytes()
                assert numerical_state((reset/'state.bin').read_bytes())==numerical_state((first/'state.bin').read_bytes())
                # The envelope before the embedded core state contains config,
                # RNG, turn count and every history token: reset must restore it.
                reset_snapshot=(b/'kv/chat.mcs').read_bytes()
                prefix=first_snapshot.index(b'MCSKV001')
                assert reset_snapshot[:prefix]==first_snapshot[:prefix]
                # Checksum passes only when the exact current payload matches it.
                snap=b/'kv/chat.mcs';old=snap.read_bytes();snap.write_bytes(old[:-2]);step(model,cache,b.name,'',rc=1);snap.write_bytes(old)
                poisoned=bytearray(old);poisoned[12]^=1;poisoned[-64:]=hashlib.sha256(poisoned[:-64]).hexdigest().encode()
                snap.write_bytes(poisoned);step(model,cache,b.name,'',rc=1);snap.write_bytes(old)
            p=run([cli,'--chat',model,'--context','256'],'hello\n:exit\n'.encode())
            directory=Path(re.search(rb'\[Chat\] Directory: ([^\r\n]+)',p.stdout)[1].decode())
            assert (directory/'kv/chat.mcs').is_file()
            run([cli,'--chat',model,'--context','256','--session',directory.name],b'next\n:exit\n')
            listed=run([cli],f'load "{model}"\nchat list\nexit\n'.encode())
            assert directory.name.encode() in listed.stdout
    print('Cross-process greedy/stochastic continuation, reset RNG, corrupt/wrong model/shape rejection and CLI: PASS')
if __name__=='__main__':main()
