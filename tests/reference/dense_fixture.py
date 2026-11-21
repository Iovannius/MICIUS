"""Independent packing and scalar reference for dense GGUF extensions.

Build packed blocks from logical codes/scales, rather than reimplementing the
production bit decoder. Q4_K_M is a tensor selection recipe, not a GGML type.
"""
import math
import random
import struct
from pathlib import Path

def f32(x): return struct.unpack('<f',struct.pack('<f',x))[0]
def string(s):
    b=s.encode(); return struct.pack('<Q',len(b))+b

def block(kind,rng):
    if kind==12:
        d,dm=2**-12,2**-13
        scales=[rng.randrange(64) for _ in range(8)]
        mins=[rng.randrange(64) for _ in range(8)]
        codes=[rng.randrange(16) for _ in range(256)]
        s=bytearray(12)
        for j in range(4):
            s[j]=scales[j]|((scales[j+4]>>4)<<6)
            s[j+4]=mins[j]|((mins[j+4]>>4)<<6)
            s[j+8]=(scales[j+4]&15)|((mins[j+4]&15)<<4)
        q=bytes(codes[p*64+j]|(codes[p*64+32+j]<<4) for p in range(4) for j in range(32))
        raw=struct.pack('<ee',d,dm)+s+q
        vals=[f32(f32(d*scales[i//32])*codes[i]-f32(dm*mins[i//32])) for i in range(256)]
    elif kind==14:
        d=2**-12
        scales=[rng.randrange(-16,17) for _ in range(16)]
        codes=[rng.randrange(-32,32) for _ in range(256)]
        lo=bytearray(128);hi=bytearray(64)
        for i,c in enumerate(codes):
            h=i//128;s=(i%128)//32;j=i%32;q=c+32
            lo[h*64+(s%2)*32+j]|=(q&15)<<(4*(s//2))
            hi[h*32+j]|=(q>>4)<<(2*s)
        raw=lo+hi+struct.pack('<16be',*scales,d)
        vals=[f32(f32(d*scales[i//16])*c) for i,c in enumerate(codes)]
    elif kind==8:
        d=2**-8;codes=[rng.randrange(-16,17) for _ in range(32)]
        raw=struct.pack('<e32b',d,*codes);vals=[d*c for c in codes]
    else:
        v=rng.uniform(-.06,.06)
        raw=struct.pack('<e' if kind==1 else '<f',v)
        vals=[struct.unpack('<e' if kind==1 else '<f',raw)[0]]
    return bytes(raw),vals

def write_gguf(path,meta,tensors):
    info=bytearray();payload=bytearray()
    for name,(dims,kind,raw) in tensors.items():
        payload+=b'\0'*(-len(payload)%32)
        info+=string(name)+struct.pack('<I',len(dims))+struct.pack('<'+'Q'*len(dims),*dims)+struct.pack('<IQ',kind,len(payload))
        payload+=raw
    records=b''.join(string(k)+struct.pack('<I',t)+v for k,(t,v) in meta.items())
    header=b'GGUF'+struct.pack('<IQQ',3,len(tensors),len(meta))+records+info
    Path(path).write_bytes(header+b'\0'*(-len(header)%32)+payload)

def fixture(path,arch='qwen3',mixed=True,tied=False,rope=True,vocab=64,context=128):
    hidden,heads,kvheads,hd,ffn=256,8 if arch=='qwen3' else 4,2,64,256
    qdim,kdim=heads*hd,kvheads*hd
    rng=random.Random(7331);meta={};tensors={};weights={}
    def md(k,t,v): meta[k]=(t,v)
    md('general.architecture',8,string(arch));md('general.name',8,string('Independent third-party synthetic model'))
    md('general.file_type',4,struct.pack('<I',15 if mixed else 7))
    md('tokenizer.ggml.model',8,string('none'))
    md('tokenizer.ggml.tokens',9,struct.pack('<IQ',8,vocab)+b''.join(string(str(i)) for i in range(vocab)))
    for k,v in {'embedding_length':hidden,'block_count':2,'context_length':context,'attention.head_count':heads,
                'attention.head_count_kv':kvheads,'attention.key_length':hd,'attention.value_length':hd,
                'feed_forward_length':ffn,'rope.dimension_count':hd}.items():
        md(arch+'.'+k,4,struct.pack('<I',v))
    for k,v in {'rope.freq_base':10000.,'attention.layer_norm_rms_epsilon':1e-6}.items(): md(arch+'.'+k,6,struct.pack('<f',v))
    def vector(name,n,values=None,kind=0):
        vals=values or [f32(rng.uniform(.75,1.25)) for _ in range(n)]
        raw=struct.pack('<'+('e' if kind==1 else 'f')*n,*vals)
        tensors[name]=([n],kind,raw);weights[name]=[x[0] for x in struct.iter_unpack('<e' if kind==1 else '<f',raw)]
    def matrix(name,cols,rows,kind):
        raw=bytearray();vals=[];bs=256 if kind in (12,14) else 32 if kind==8 else 1
        for _ in range(cols*rows//bs):
            r,v=block(kind,rng);raw+=r;vals+=v
        tensors[name]=([cols,rows],kind,bytes(raw));weights[name]=[vals[i:i+cols] for i in range(0,len(vals),cols)]
    matrix('token_embd.weight',hidden,vocab,14 if mixed else 8)
    if not tied:matrix('output.weight',hidden,vocab,14 if mixed else 8)
    vector('output_norm.weight',hidden)
    if arch=='llama' and rope:
        md('llama.rope.scaling.type',8,string('llama3'))
        vector('rope_freqs.weight',hd//2,[f32(1+j*.125) for j in range(hd//2)])
    for layer in range(2):
        p=f'blk.{layer}.'
        vector(p+'attn_norm.weight',hidden);vector(p+'ffn_norm.weight',hidden,kind=1)
        for j,(name,c,r) in enumerate([('attn_q',hidden,qdim),('attn_k',hidden,kdim),('attn_v',hidden,kdim),
            ('attn_output',qdim,hidden),('ffn_gate',hidden,ffn),('ffn_up',hidden,ffn),('ffn_down',ffn,hidden)]):
            matrix(p+name+'.weight',c,r,(12 if j%2==0 else 14) if mixed else 8)
        if arch=='qwen3':
            vector(p+'attn_q_norm.weight',hd);vector(p+'attn_k_norm.weight',hd)
        if arch=='qwen2':
            for name,n in [('attn_q',qdim),('attn_k',kdim),('attn_v',kdim)]:vector(p+name+'.bias',n,[f32(rng.uniform(-.3,.3)) for _ in range(n)])
    write_gguf(path,meta,tensors)
    return meta,tensors,weights,(hidden,heads,kvheads,hd)

def forward(w,shape,tokens,arch,disable_norm=False,disable_factors=False):
    hidden,heads,kvheads,hd=shape
    def norm(x,scale):
        inv=1/math.sqrt(sum(v*v for v in x)/len(x)+f32(1e-6))
        return [f32(v*inv*s) for v,s in zip(x,scale)]
    def project(name,x):return [f32(sum(a*b for a,b in zip(row,x))) for row in w[name]]
    def add(x,y):return [f32(a+b) for a,b in zip(x,y)]
    def rotate(x,pos):
        y=x[:]
        for base in range(0,len(x),hd):
            for j in range(hd//2):
                theta=f32(f32(pos)/f32(10000.**f32(2*j/hd)))
                if 'rope_freqs.weight' in w and not disable_factors:theta=f32(theta/w['rope_freqs.weight'][j])
                c,s=f32(math.cos(theta)),f32(math.sin(theta))
                a=base+(2*j if arch=='llama' else j);b=a+(1 if arch=='llama' else hd//2)
                y[a]=f32(f32(x[a]*c)-f32(x[b]*s));y[b]=f32(f32(x[b]*c)+f32(x[a]*s))
        return y
    cache=[([],[]) for _ in range(2)];result=[]
    for pos,token in enumerate(tokens):
        x=w['token_embd.weight'][token]
        for layer,(keys,values) in enumerate(cache):
            p=f'blk.{layer}.';n=norm(x,w[p+'attn_norm.weight'])
            q,k,v=[project(p+'attn_'+name+'.weight',n) for name in ('q','k','v')]
            if arch=='qwen2':q,k,v=[add(a,w[p+'attn_'+name+'.bias']) for a,name in zip((q,k,v),('q','k','v'))]
            if arch=='qwen3' and not disable_norm:
                q=[a for i in range(0,len(q),hd) for a in norm(q[i:i+hd],w[p+'attn_q_norm.weight'])]
                k=[a for i in range(0,len(k),hd) for a in norm(k[i:i+hd],w[p+'attn_k_norm.weight'])]
            q=rotate(q,pos);keys.append(rotate(k,pos));values.append(v);attn=[]
            for head in range(heads):
                offset=head//(heads//kvheads)*hd;query=q[head*hd:(head+1)*hd]
                scores=[sum(a*b for a,b in zip(query,key[offset:offset+hd]))/math.sqrt(hd) for key in keys]
                probs=[math.exp(f32(s)-max(scores)) for s in scores];total=sum(probs)
                probs=[f32(v)/total for v in probs]
                attn.extend(f32(sum(row[offset+d]*prob for row,prob in zip(values,probs))) for d in range(hd))
            x=add(x,project(p+'attn_output.weight',attn));n=norm(x,w[p+'ffn_norm.weight'])
            gate=project(p+'ffn_gate.weight',n);up=project(p+'ffn_up.weight',n)
            act=[f32(g/(1+math.exp(-g))*u) for g,u in zip(gate,up)]
            x=add(x,project(p+'ffn_down.weight',act))
        result.extend(project('output.weight' if 'output.weight' in w else 'token_embd.weight',norm(x,w['output_norm.weight'])))
    return result
