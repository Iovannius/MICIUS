"""Known template profiles, byte-BPE boundaries and generic-path persisted KV."""
from pathlib import Path
import hashlib,json,struct,subprocess,sys,tempfile
from chat_text_tests import alphabet
from reference.dense_fixture import fixture,string,write_gguf

def metadata(profile,template):
    words=alphabet()+['ab','bc','abc','aa','aaaa',alphabet()[32]+'1','12','!!']
    controls=['<|begin_of_text|>','<|start_header_id|>','<|end_header_id|>','<|eot_id|>','<|end_of_text|>'] if profile=='llama-bpe' else ['<|im_start|>','<|im_end|>','<|endoftext|>']
    kinds=[1]*len(words)+[3]*len(controls);words+=controls
    if profile=='qwen2':words+=['<think>','</think>'];kinds+=[4,4]
    while len(words)<288:kinds+=[5];words+=['unused'+str(len(words))]
    def array(t,x):return struct.pack('<IQ',t,len(x))+b''.join(x)
    values={'tokenizer.ggml.model':(8,string('gpt2')),'tokenizer.ggml.pre':(8,string(profile)),
        'tokenizer.ggml.tokens':(9,array(8,list(map(string,words)))),
        'tokenizer.ggml.token_type':(9,array(5,[struct.pack('<i',i) for i in kinds])),
        'tokenizer.ggml.merges':(9,array(8,list(map(string,['a b','b c','ab c','a a','aa aa',alphabet()[32]+' 1','1 2','! !'])))),
        'tokenizer.ggml.eos_token_id':(4,struct.pack('<I',267 if profile=='llama-bpe' else 265)),
        'tokenizer.ggml.add_bos_token':(7,bytes([profile=='llama-bpe'])),
        'tokenizer.chat_template':(8,string(template))}
    if profile=='llama-bpe':values['tokenizer.ggml.bos_token_id']=(4,struct.pack('<I',264))
    return values

def main():
    tok,prompt,session=map(lambda s:Path(s).resolve(),sys.argv[1:])
    templates=json.loads((Path(__file__).parent/'text/dense-templates.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(templates['Qwen-Qwen3-0.6B-slice-filters'].encode()).hexdigest()=='5595e8741bb5500f9d395e05a0aefe6ca1e8d54c5dd8b98bcf7760d895fdc6e6'
    qwen_prompts={}
    with tempfile.TemporaryDirectory(dir=Path.cwd(),prefix='dense-text-') as tmp:
        root=Path(tmp);inp=root/'input';out=root/'out';index=0
        def run(cmd,ok=True):
            p=subprocess.run(list(map(str,cmd)),capture_output=True,timeout=90)
            assert (p.returncode==0)==ok,(cmd,p.returncode,p.stderr[-1000:]);return p
        for name,template in templates.items():
            llama=name.startswith('meta-llama');arch='llama' if llama else 'qwen3';pre='llama-bpe' if llama else 'qwen2'
            path=root/(name+'.gguf');m,t,_,_=fixture(path,arch,vocab=288,context=512)
            m.update(metadata(pre,template));t['output.weight']=([256,288],0,bytes(256*288*4))
            write_gguf(path,m,t)
            inp.write_bytes(b'1234');run([tok,path,inp,out]);ids=list(struct.unpack('<'+'I'*(out.stat().st_size//4),out.read_bytes()))
            assert ids==([262,51,52] if llama else [49,50,51,52]),ids
            inp.write_text(' 你好\u3000',encoding='utf-8')
            for turn in ['first','next']:
                run([prompt,'--prompt-only',path,inp,turn,out]);ids=list(struct.unpack('<'+'I'*(out.stat().st_size//4),out.read_bytes()))
                if llama:
                    def head(s):return [265,*s.encode(),266,10,10]
                    expected=([264,*head('system'),*b'Cutting Knowledge Date: December 2023\nToday Date: 26 Jul 2024\n\n',267] if turn=='first' else [])
                    expected+=head('user')+list('你好'.encode())+[267]+head('assistant')
                else:expected=[264,*'user\n 你好\u3000'.encode(),265,10,264,*b'assistant\n',267,10,10,268,10,10]
                assert ids==expected,(name,turn,ids,expected)
            if not llama:
                for text in ('Reply with only the word blue.','刚才说的颜色是什么？ Reply with one word.',
                             '','<tool_response>','<tool_response>你好</tool_response>',
                             'one\ntwo','<think>text</think>',' 你好\u3000'):
                    inp.write_text(text,encoding='utf-8')
                    for turn in ('first','next'):
                        run([prompt,'--prompt-only',path,inp,turn,out])
                        key=(text,turn);value=out.read_bytes()
                        assert value==qwen_prompts.setdefault(key,value),'Qwen3 template variant changed prompt tokens'
            if name not in ('Qwen-Qwen3-0.6B','Qwen-Qwen3-0.6B-slice-filters','meta-llama-Llama-3.1-8B-Instruct'):continue
            # Use all-zero LM-head to keep arbitrary model output printable/valid UTF-8 (NUL).
            def step(id,lines):
                nonlocal index
                index+=1;d=root/str(index);d.mkdir();(d/'input').write_text(lines)
                conf=root/'sampling.conf';conf.write_text('version=1\nmax_new_tokens=2\n')
                cmd=[session,path,root/'cache',512,id,d/'input',d]
                if id=='-':cmd+=[conf]
                run(cmd);return d,Path((d/'directory.txt').read_text())
            whole,a=step('-','first\nnext\n');first,b=step('-','first\n');resume,b=step(b.name,'next\n')
            assert (whole/'state.bin').read_bytes()==(resume/'state.bin').read_bytes()
            assert (a/'kv/chat.mcs').read_bytes()==(b/'kv/chat.mcs').read_bytes()
            assert (whole/'turn-2.utf8').read_bytes()==(resume/'turn-1.utf8').read_bytes()
            # Unsupported templates must not silently use the profile above.
            m['tokenizer.chat_template']=(8,string('custom unsupported'));write_gguf(path,m,t)
            run([prompt,'--prompt-only',path,inp,'first',out],ok=False)
            m['tokenizer.chat_template']=(8,string(template+' '));write_gguf(path,m,t)
            run([prompt,'--prompt-only',path,inp,'first',out],ok=False)
    print('Dense byte-BPE/template profiles and mixed-weight cross-process snapshots: PASS')
if __name__=='__main__':main()
