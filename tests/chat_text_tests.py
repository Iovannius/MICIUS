"""Small fixtures only. Exercise real Shell I/O and cross-check incremental KV against replay."""
from pathlib import Path
import json
import os
import re
import struct
import subprocess
import sys
import tempfile

from reference.make_fixture import make_fixture
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from m2_run import summarize_chat

def alphabet():
    present={*range(33,127),*range(161,173),*range(174,256)}
    extra=iter(range(256,512))
    return [chr(b if b in present else next(extra)) for b in range(256)]

def metadata(profile, template, *, missing_byte=False):
    words=alphabet()
    ordinary=words+["ab","bc","abc","aa","aaaa",words[32]+"1","12","!!"]
    if missing_byte: ordinary[32]="badspace"
    words=ordinary+["<|im_start|>","<|im_end|>","<|endoftext|>","<tool_call>"]
    types=[1]*len(ordinary)+[3,3,3,4]
    while len(words)<288:
        types.append(5); words.append("unused"+str(len(words)))
    def string(s):
        b=s.encode();return struct.pack("<Q",len(b))+b
    def array(typ,items):
        return struct.pack("<IQ",typ,len(items))+b"".join(items)
    ranks=["a b","b c","ab c","a a","aa aa",alphabet()[32]+" 1","1 2","! !"]
    return [("tokenizer.ggml.model",8,string("gpt2")),("tokenizer.ggml.pre",8,string(profile)),
            ("tokenizer.ggml.tokens",9,array(8,[string(s) for s in words])),
            ("tokenizer.ggml.merges",9,array(8,[string(s) for s in ranks])),
            ("tokenizer.ggml.token_type",9,array(5,[struct.pack("<i",i) for i in types])),
            ("tokenizer.ggml.eos_token_id",4,struct.pack("<I",len(ordinary)+1)),
            ("tokenizer.ggml.add_bos_token",7,b"\x00"),("tokenizer.chat_template",8,string(template))]

def controlled(name,dims,kind,raw):
    # Identity K/V makes the cache sensitive to token identity and position.
    # All residual updates are zero; the LM head maps newline -> E4 BD A0 -> EOS.
    states={10:0,0xe4:1,0xbd:2,0xa0:3}
    if name=="token_embd.weight":
        width,rows=dims;data=bytearray(len(raw))
        for row in range(rows):
            struct.pack_into("<f",data,(row*width+states.get(row,4+row%(width-4)))*4,1.)
        return data
    if kind==0 and name.endswith(".bias"):
        return bytes(len(raw))
    if kind!=8:return raw
    cols,rows=dims
    block=struct.pack("<e",1.)+bytes(32)
    data=bytearray(block*(cols//32*rows))
    def weight(row,col):
        data[(row*(cols//32)+col//32)*34+2+col%32]=1
    if name=="output.weight":
        for col,row in enumerate([0xe4,0xbd,0xa0,265]):weight(row,col)
    elif name.endswith(("attn_k.weight","attn_v.weight")):
        for row in range(rows):weight(row,row%cols)
    return data

def main():
    probe,contract,cli=(Path(x).resolve() for x in sys.argv[1:])
    templates=json.loads((Path(__file__).parent/"text/templates.json").read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
        d=Path(tmp); env=dict(os.environ,MICIUS_CACHE_ROOT=str(d/"cache"))
        def run(args,stdin=b"",rc=0):
            p=subprocess.run([str(x) for x in args],input=stdin,cwd=d,env=env,capture_output=True,timeout=60)
            assert p.returncode==rc,(args,p.returncode,p.stdout[-1400:],p.stderr[-1400:])
            return p
        def fixture(path,profile,template,**opts):
            return make_fixture(d/path,context=384,vocab=288,architecture="llama" if profile=="smollm" else "qwen2",
                                tokenizer_metadata=metadata(profile,template,**opts),tensor_override=controlled)
        for profile,sysmsg in (("smollm","You are a helpful AI assistant named SmolLM, trained by Hugging Face"),
                              ("qwen2","You are Qwen, created by Alibaba Cloud. You are a helpful assistant.")):
            model=fixture(profile+" ok.gguf",profile,templates[profile])
            def ids(p): return [v[0] for v in struct.iter_unpack("<I",p.read_bytes())]
            inp=d/"input.bin";out=d/"out.u32"
            for s,expected in [("abc",[258]),("aaaaa",[260,97]),(" 12",[32,49,50]),("<tool_call>",[267]),
                               ("你好🌍\x00",list("你好🌍\x00".encode())),("",[]),("<|im_start|>",list(b"<|im_start|>"))]:
                inp.write_bytes(s.encode());run([probe,model,inp,out]);assert ids(out)==expected,(s,ids(out),expected)
            inp.write_bytes(b"\xc0\xbf");run([probe,model,inp,out],rc=1)
            inp.write_bytes(" 你好🙂".encode())
            for turn in ("first","next"):
                run([contract,"--prompt-only",model,inp,turn,out])
                sys_ids=list(("system\n"+sysmsg).encode())
                if profile=="qwen2":
                    # The synthetic rank-zero "a b" merge occurs once in Alibaba.
                    where=("system\n"+sysmsg).index("ab")
                    sys_ids[where:where+2]=[256]
                expected=([264,*sys_ids,265,10] if turn=="first" else [])
                expected.extend([264,*"user\n 你好🙂".encode(),265,10,264,*b"assistant\n"])
                assert ids(out)==expected
            run([contract,model])
            args=[cli,"--chat",model,"--context","256","--max-new","8","--trace-turns"]
            test=run(args,"你好\n再说一次\n:reset\n你好\n:exit\n".encode())
            assert test.stdout.count("你".encode())==3 and test.stdout.count(b"Assistant: ")==3
            trace=re.findall(rb'\[chat-turn\] turn=(\d+) finish=eot tokens=3 context=(\d+)', test.stderr)
            assert len(trace)==3 and trace[0]==trace[2] and trace[1][0]==b"2" and int(trace[1][1])>int(trace[0][1])
            assert b"[chat-reset]" in test.stderr
            reviewed=summarize_chat(test.stdout,test.stderr,8)
            assert [r["text"] for r in reviewed]==["你"]*3
            for stdout,stderr in [(test.stdout,test.stderr.replace(b"[chat-reset]",b"")),
                                  (test.stdout.replace("你".encode(),b"x",1),test.stderr),
                                  (test.stdout,test.stderr+b"[chat-error]\n")]:
                try: summarize_chat(stdout,stderr,8)
                except ValueError: pass
                else: raise AssertionError("Corrupt native transcript passed")
            directlog=d/(profile+"-direct.txt")
            run(args,(':log start "'+directlog.name+'"\nhello\n:log stop\nnot logged\n:exit\n').encode())
            saved=directlog.read_text(encoding="utf-8")
            assert "You: hello\nAssistant: 你\n" in saved and "not logged" not in saved
            # The public numeric shell and append-only opt-in transcript coexist.
            log=d/("session 日志 "+profile+".txt");log.write_text("original\n",encoding="utf-8")
            commands=[ 'log start "absent/file"', 'log start "'+log.name+'"', 'load "'+model.name+'"',
                       "chat 8","  hello  世界!",":exit","log stop","stats","exit",""]
            logged=run([cli],"\n".join(commands).encode())
            text=log.read_text(encoding="utf-8")
            assert text.startswith("original\n") and "You:   hello  世界!\n" in text
            assert text.count("Assistant: 你\n")==1 and "\x1b" not in text and "Runtime Telemetry" not in text
            assert b"Failed to open log file" in logged.stdout
            bad=fixture(profile+" bad.gguf",profile,"unknown template")
            fail=run([cli,"--chat",bad,"--context","128"],rc=1); assert b"template" in fail.stderr
            raw=run([cli],('load "'+bad.name+'"\ngenerate 10\nexit\n').encode());assert b"Prefill complete" in raw.stdout
            missing=fixture(profile+" bad-coverage.gguf",profile,templates[profile],missing_byte=True)
            run([probe,missing,inp,out],rc=1)
    print("Byte BPE, literal control spelling, UTF-8 fragments, two templates, multi-turn KV, cancel/retry, EOT, reset, cap, logs and numeric fallback: PASS")

if __name__=="__main__":main()
