"""Independent scalar Qwen2 oracle for the fixed small GGUF, with negative controls."""
import hashlib
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from reference.make_fixture import make_fixture

TOKENS = (3, 61, 7, 11, 2, 17)
HEAD = 32
HIDDEN = 64

def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]

def add(a, b):
    return [f32(x + y) for x, y in zip(a, b)]

def norm(x):
    factor = (sum(v*v for v in x)/len(x)+f32(1e-6))**-0.5
    return [f32(v*factor) for v in x]

def oracle(tensors, backend, *, split=True, bias=True):
    matrices, vectors = {}, {}
    for name, (dims, kind, raw) in tensors.items():
        if kind == 0:
            vectors[name] = [v for v, in struct.iter_unpack("<f", raw)]
        else:
            blocks = list(struct.iter_unpack("<e32b", raw))
            width = dims[0]//32
            matrices[name] = [blocks[i:i+width] for i in range(0,len(blocks),width)]
    def project(name, x):
        chunks = [x[i:i+32] for i in range(0,len(x),32)]
        if backend == "INT8":
            activations = []
            for chunk in chunks:
                d = max(map(abs,chunk))/127.0
                activations.append((d, [round(v/d) if d else 0 for v in chunk]))
        else:
            activations = [(1.0,chunk) for chunk in chunks]
        return [f32(sum(w[0]*sum(c*v for c,v in zip(w[1:],vals))*d
                        for w,(d,vals) in zip(row,activations))) for row in matrices[name]]
    def rotate(x, position):
        y = x[:]
        for offset in range(0,len(x),HEAD):
            for j in range(HEAD//2):
                i = offset + (j if split else 2*j)
                k = i + (HEAD//2 if split else 1)
                theta = position / (1e6 ** (2*j/HEAD))
                cs, sn = f32(math.cos(theta)), f32(math.sin(theta))
                y[i], y[k] = f32(x[i]*cs-x[k]*sn), f32(x[i]*sn+x[k]*cs)
        return y
    caches = [([],[]) for _ in range(2)]
    output = []
    for pos,token in enumerate(TOKENS):
        x = vectors["token_embd.weight"][token*HIDDEN:(token+1)*HIDDEN]
        for layer,(keys,values) in enumerate(caches):
            prefix = f"blk.{layer}."
            normalized = norm(x)
            def projection(kind):
                result = project(prefix+kind+".weight",normalized)
                return add(result,vectors[prefix+kind+".bias"]) if bias else result
            q = rotate(projection("attn_q"),pos)
            keys.append(rotate(projection("attn_k"),pos))
            values.append(projection("attn_v"))
            heads = []
            for offset in range(0,HIDDEN,HEAD):
                query = q[offset:offset+HEAD]
                scores = [sum(a*b for a,b in zip(query,key))/math.sqrt(HEAD) for key in keys]
                p = [math.exp(f32(v)-max(scores)) for v in scores]
                weights = [f32(v)/sum(p) for v in p]
                heads.extend(f32(sum(value[d]*w for value,w in zip(values,weights))) for d in range(HEAD))
            x = add(x,project(prefix+"attn_output.weight",heads))
            normalized = norm(x)
            gate = project(prefix+"ffn_gate.weight",normalized)
            up = project(prefix+"ffn_up.weight",normalized)
            act = [f32(g/(1+math.exp(-g))*u) for g,u in zip(gate,up)]
            x = add(x,project(prefix+"ffn_down.weight",act))
        output.extend(project("output.weight",norm(x)))
    return output

def main():
    exe, backend = Path(sys.argv[1]).resolve(), sys.argv[2]
    with tempfile.TemporaryDirectory(dir=Path.cwd(),prefix="qwen2-fixture-") as d:
        root = Path(d)
        cache = root/"cache"
        tensors = {}
        model = make_fixture(root/"model.gguf", head_dim=HEAD, architecture="qwen2", tensor_data=tensors)
        seq, out = root/"ids", root/"logits"
        seq.write_bytes(struct.pack("<6I",*TOKENS))
        def probe(path, cache_path=cache):
            return subprocess.run([str(exe),str(path),str(cache_path),str(seq),str(out)],capture_output=True,text=True)
        r = probe(model)
        assert r.returncode==0, r.stdout+r.stderr
        info = dict(line.split("=",1) for line in r.stdout.splitlines() if "=" in line)
        context = Path(info["context_directory"])
        assert info["architecture"]=="qwen2" and info["reset_replay"]=="PASS"
        model_id=hashlib.sha256(model.read_bytes()).hexdigest()
        assert context.parent == cache/model_id/"contexts" and (context/"kv").is_dir()
        actual = [v for v, in struct.iter_unpack("<f",out.read_bytes())]
        expected = oracle(tensors,backend)
        assert len(actual)==len(expected)==len(TOKENS)*128
        error = max(abs(a-b) for a,b in zip(actual,expected))
        assert error < 2e-4, (backend,error)
        for wrong in (oracle(tensors,backend,split=False),oracle(tensors,backend,bias=False)):
            assert max(abs(a-b) for a,b in zip(expected,wrong))>0.01
        # Same file, new session. No output or KV record is shared.
        again=probe(model)
        assert again.returncode==0 and len(list(context.parent.iterdir()))==2
        other=root/"other/model.gguf"
        make_fixture(other, head_dim=HEAD, architecture="qwen2", vocab=127)
        assert probe(other).returncode==0 and len(list(cache.iterdir()))==2
        before=sorted(cache.rglob("*"))
        bad=make_fixture(root/"bad.gguf",architecture="qwen2",omit_tensor="blk.1.attn_k.bias")
        failure=probe(bad)
        assert failure.returncode and "attn_k.bias" in failure.stderr
        assert sorted(cache.rglob("*"))==before
        obstruction=root/"file-as-cache";obstruction.write_text("keep")
        assert probe(model,obstruction).returncode and obstruction.read_text()=="keep"
        print(f"{backend}: independent Qwen2 logits max absolute error {error:.8g}, bias and RoPE controls passed")

if __name__=="__main__":
    main()
