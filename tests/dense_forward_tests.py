"""No downloaded model inference: independent packed GGUF integration checks."""
from pathlib import Path
import math,random,struct,subprocess,sys,tempfile
from reference.dense_fixture import block,fixture,forward,string,write_gguf

def main():
    model_probe,quant_probe=map(lambda x:Path(x).resolve(),sys.argv[1:3])
    with tempfile.TemporaryDirectory(dir=Path.cwd(),prefix='dense-contract-') as temp:
        root=Path(temp);inp=root/'input';out=root/'out';seq=root/'seq'
        for kind in (0,1,8,12,14):
            rng=random.Random(kind+733);raw=bytearray();expected=[]
            for _ in range(20):r,v=block(kind,rng);raw+=r;expected+=v
            inp.write_bytes(raw)
            cmd=[str(quant_probe),str(kind),str(len(expected)),str(inp),str(out)]
            assert subprocess.run(cmd).returncode==0
            assert out.read_bytes()==struct.pack('<'+'f'*len(expected),*expected),(kind,'decode mismatch')
            inp.write_bytes(raw[:-1]);assert subprocess.run(cmd).returncode!=0
        inp.write_bytes(struct.pack('<f',float('nan')))
        assert subprocess.run([str(quant_probe),'0','1',str(inp),str(out)]).returncode!=0
        tokens=[3,21,7,11,2,17];seq.write_bytes(struct.pack('<6I',*tokens))
        def run(path):return subprocess.run([str(model_probe),str(path),str(root/'cache'),str(seq),str(out)],capture_output=True,text=True)
        cases=[('qwen3',True,False),('llama',True,False),('qwen2',True,True),('qwen3',False,False)]
        for arch,mixed,tied in cases:
            model=root/f'{arch}-{mixed}-{tied}.gguf';meta,tensors,w,shape=fixture(model,arch,mixed,tied)
            r=run(model);assert r.returncode==0,r.stdout+r.stderr
            actual=struct.unpack('<'+'f'*(out.stat().st_size//4),out.read_bytes());expected=forward(w,shape,tokens,arch)
            assert len(actual)==len(expected)
            err=max(abs(a-b) for a,b in zip(actual,expected))
            assert err<3e-4,(arch,mixed,err)
            if arch in ('llama','qwen3'):
                wrong=forward(w,shape,tokens,arch,disable_norm=arch=='qwen3',disable_factors=arch=='llama')
                assert max(abs(a-b) for a,b in zip(expected,wrong))>.001
            print(arch,'mixed' if mixed else 'Q8','max abs error',err)
        # Tied non-Q8 embeddings are also the output matrix. Exercise dispatch
        # without another non-Q8 matrix or a Qwen3 feature forcing the route.
        for kind in (0,1):
            model=root/f'tied-{kind}.gguf'
            meta,tensors,w,shape=fixture(model,'llama',False,True,False)
            vals=[v for row in w['token_embd.weight'] for v in row]
            raw=struct.pack('<'+('f' if kind==0 else 'e')*len(vals),*vals)
            tensors['token_embd.weight']=([256,64],kind,raw)
            write_gguf(model,meta,tensors)
            r=run(model);assert r.returncode==0,r.stdout+r.stderr
            actual=struct.unpack('<'+'f'*(out.stat().st_size//4),out.read_bytes())
            assert max(abs(a-b) for a,b in zip(actual,forward(w,shape,tokens,'llama')))<3e-4
        # Malformed or semantically unsupported models are rejected before a context is published.
        meta,tensors,_,_=fixture(root/'base.gguf','qwen3')
        before=set((root/'cache').rglob('*'))
        def reject(m,t):
            write_gguf(root/'bad.gguf',m,t);r=run(root/'bad.gguf');assert r.returncode!=0,r.stdout
            assert set((root/'cache').rglob('*'))==before,'failed load published context'
        t=dict(tensors);del t['blk.0.attn_q_norm.weight'];reject(meta,t)
        t=dict(tensors);dims,kind,raw=t['blk.0.attn_k_norm.weight'];t['blk.0.attn_k_norm.weight']=([32],kind,raw[:128]);reject(meta,t)
        t=dict(tensors);dims,kind,raw=t['blk.0.attn_q.weight'];t['blk.0.attn_q.weight']=([128,1024],kind,raw);reject(meta,t)
        t=dict(tensors);dims,kind,raw=t['output_norm.weight'];t['output_norm.weight']=(dims,kind,struct.pack('<f',float('inf'))+raw[4:]);reject(meta,t)
        for field,value in [('expert_count',2),('attention.sliding_window',128),('attention.value_length',32)]:
            m=dict(meta);m['qwen3.'+field]=(4,struct.pack('<I',value));reject(m,tensors)
        for field in ('logit_scale','embedding_scale','residual_scale','attn_logit_softcapping',
                      'final_logit_softcapping','attention.clamp_kqv','attention.scale','rope.scaling.factor'):
            m=dict(meta);m['qwen3.'+field]=(6,struct.pack('<f',2.0));reject(m,tensors)
        m=dict(meta);m['qwen3.rope.scaling.type']=(8,string('yarn'));reject(m,tensors)
        for arch in ('qwen3moe','qwen35','llama4'):
            m=dict(meta);m['general.architecture']=(8,string(arch));reject(m,tensors)
        m,t,_,_=fixture(root/'llama.gguf','llama');del t['rope_freqs.weight'];reject(m,t)
        print('Independent Q4_K/Q6_K packing, dense forward, reset replay and rejection checks: PASS')
if __name__=='__main__':main()
