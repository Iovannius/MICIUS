"""Generate a deterministic tiny Llama GGUF for integration checks, not quality/performance."""
import argparse
from pathlib import Path
import random
import struct

def make_fixture(path, context=256, head_dim=32, heads=2, kv_heads=1, q8_embedding=False, tied=False, vocab=128,
                 architecture="llama", tensor_data=None, omit_tensor=None, tokenizer_metadata=None,
                 tensor_override=None):
    assert head_dim in (32,64)
    assert heads>0 and kv_heads>0 and heads%kv_heads==0
    hidden=heads*head_dim; kv=kv_heads*head_dim; ffn=3*head_dim
    if architecture == "qwen2": ffn = 5*hidden
    rng = random.Random(1003)
    def u32(x): return struct.pack('<I', x)
    def u64(x): return struct.pack('<Q', x)
    def string(x):
        b=x.encode(); return u64(len(b))+b
    meta=[]
    def md(k,t,b): meta.append(string(k)+u32(t)+b)
    md('general.architecture',8,string(architecture))
    if tokenizer_metadata is None: md('tokenizer.ggml.model',8,string('none'))
    md(architecture+'.vocab_size',4,u32(vocab))
    md('general.name',8,string('MICIUS I1 synthetic integration fixture'))
    if tokenizer_metadata is None:
        md('tokenizer.ggml.tokens',9,u32(8)+u64(vocab)+b''.join(string(str(i)) for i in range(vocab)))
    else:
        for key, kind, data in tokenizer_metadata: md(key,kind,data)
    for k,v in {'embedding_length':hidden,'context_length':context,'block_count':2,
                'attention.head_count':heads,'attention.head_count_kv':kv_heads,
                'feed_forward_length':ffn,'rope.dimension_count':head_dim}.items():
        if architecture == 'qwen2' and k == 'rope.dimension_count': continue
        md(architecture+'.'+k,4,u32(v))
    md('general.alignment',4,u32(32))
    for k,v in [('attention.layer_norm_rms_epsilon',1e-6 if architecture=='qwen2' else 1e-5),('rope.freq_base',1e6 if architecture=='qwen2' else 10000.)]:
        md(architecture+'.'+k,6,struct.pack('<f',v))
    tensors=[]
    def f32(name,dims,values):
        tensors.append((name,dims,0,struct.pack('<'+'f'*len(values),*values)))
    def q8(name,cols,rows):
        data=bytearray()
        for _ in range(rows*(cols//32)):
            data+=struct.pack('<e',rng.choice([0.001953125,0.00390625,0.0078125]))
            data+=struct.pack('<32b',*[rng.randrange(-32,33) for _ in range(32)])
        tensors.append((name,[cols,rows],8,bytes(data)))
    if q8_embedding:q8('token_embd.weight',hidden,vocab)
    else:f32('token_embd.weight',[hidden,vocab],[rng.uniform(-1,1) for _ in range(hidden*vocab)])
    f32('output_norm.weight',[hidden],[1.]*hidden)
    if not tied:q8('output.weight',hidden,vocab)
    for layer in range(2):
        prefix=f'blk.{layer}.'
        for name in ['attn_norm.weight','ffn_norm.weight']: f32(prefix+name,[hidden],[1.]*hidden)
        for name,cols,rows in [('attn_q',hidden,hidden),('attn_k',hidden,kv),('attn_v',hidden,kv),
                               ('attn_output',hidden,hidden),('ffn_gate',hidden,ffn),('ffn_up',hidden,ffn),('ffn_down',ffn,hidden)]:
            q8(prefix+name+'.weight',cols,rows)
        if architecture == "qwen2":
            for name,width in [("attn_q",hidden),("attn_k",kv),("attn_v",kv)]:
                f32(prefix+name+".bias",[width],[rng.uniform(-1.5,1.5) for _ in range(width)])
    if omit_tensor: tensors = [t for t in tensors if t[0] != omit_tensor]
    if tensor_data is not None: tensor_data.update({name:(dims,kind,raw) for name,dims,kind,raw in tensors})
    info=bytearray();payload=bytearray()
    for name,dims,kind,data in tensors:
        if tensor_override is not None: data=tensor_override(name,dims,kind,data)
        payload+=b'\0'*((-len(payload))%32)
        info+=string(name)+u32(len(dims))+b''.join(u64(d) for d in dims)+u32(kind)+u64(len(payload))
        payload+=data
    header=b'GGUF'+u32(3)+u64(len(tensors))+u64(len(meta))+b''.join(meta)+info
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(header+b'\0'*((-len(header))%32)+payload)
    return path

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output');a=p.parse_args()
    print(make_fixture(a.output))
