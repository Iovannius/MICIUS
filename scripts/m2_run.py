"""Build M2, verify its installed C SDK, and run the operator's bounded chat check."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid
import zipfile

from m1_prepare import ROOT, PROFILES, digest, execute, prepare, require, save
from verify_release import MANIFEST, verify

RESULTS=ROOT/"results"
USER_PROMPT=b"\x1b[36mYou: \x1b[0m"
ASSISTANT_PROMPT=b"\x1b[32mAssistant: \x1b[0m"
def read(p):
    return json.loads(p.read_bytes())

def build():
    RESULTS.mkdir(exist_ok=True)
    name="m2-"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"-"+uuid.uuid4().hex[:8]
    work=RESULTS/name
    work.mkdir()
    print("WORK",work,flush=True)
    save(work/"status.json",{"status":"PREPARING"})
    try:
        receipt=prepare(work)
        receipt["unit"]="M2"
        for b in PROFILES:
            prefix=work/"sdk"/b
            external=work/("text-consumer-"+b)
            execute(["cmake","-S",ROOT/"examples/text-consumer","-B",external,"-G","Ninja",
                     "-DCMAKE_BUILD_TYPE=Release","-DCMAKE_C_COMPILER=clang","-DCMAKE_CXX_COMPILER=clang++",
                     "-DMICIUS_DIR="+str(prefix/"lib/cmake/MICIUS")],work,work/"logs"/(b+"-text-configure.log"))
            execute(["cmake","--build",external],work,work/"logs"/(b+"-text-link.log"))
            paths={"cli":prefix/"bin/micius_cli.exe","tokenizer":external/"m2_token_probe.exe",
                   "template":work/("build-"+b)/"bin/m2_chat_contract.exe"}
            receipt["profiles"][b]["text_tools"]={k:{"path":p.relative_to(work).as_posix(),"sha256":digest(p)}
                                                   for k,p in paths.items()}
        save(work/"prepared.json",receipt)
        identity=digest(work/"prepared.json")
        save(work/"status.json",{"status":"PREPARED","prepared_sha256":identity})
        # Each publisher has its own temporary pointer when two preparations overlap.
        pointer=RESULTS/("latest-"+uuid.uuid4().hex+".tmp")
        save(pointer,{"name":name,"prepared_sha256":identity})
        os.replace(pointer,RESULTS/"M2-last-prepared.json")
    except BaseException as e:
        save(work/"status.json",{"status":"PREPARATION_STOPPED","error":str(e)})
        raise
    print("PREPARED",work,flush=True)
    return work

def validate(work):
    require(work.resolve().parent==RESULTS.resolve() and re.fullmatch(r"m2-\d{8}-\d{6}-[0-9a-f]{8}",work.name),
            "Expected a result directory belonging to this M2 package")
    verify()
    data=read(work/"prepared.json")
    require(data["unit"]=="M2" and data["status"]=="PREPARED" and
            digest(work/"prepared.json")==read(work/"status.json")["prepared_sha256"] and
            data["manifest_sha256"]==digest(MANIFEST),"Changed preparation or source package")
    bins={}
    for b in PROFILES:
        bins[b]={}
        tools=data["profiles"][b]["text_tools"]
        for k in ("cli","tokenizer","template"):
            p=(work/tools[k]["path"]).resolve()
            require(work.resolve() in p.parents and p.is_file() and digest(p)==tools[k]["sha256"],
                    f"Changed or missing {b} {k}")
            bins[b][k]=p
    return bins

def model_file(models,info):
    p=models/info["filename"]
    require(p.is_file() and p.stat().st_size==info["bytes"] and digest(p)==info["model_sha256"],
            "GGUF identity mismatch: "+str(p))
    return p

def vocab_check(folder,bins,model,text,protocol,info):
    folder.mkdir(parents=True)
    results={}
    for mode,content in (("ordinary",text),("first",protocol["first_user"]),("next",protocol["second_user"])):
        inp=folder/(mode+".utf8");out=folder/(mode+".u32");log=folder/(mode+".log")
        inp.write_bytes(content.encode("utf-8"))
        args=[bins["tokenizer"],model,inp,out] if mode=="ordinary" else [bins["template"],"--prompt-only",model,inp,mode,out]
        message=execute(args,folder,log)
        got={"sha256":digest(out),"tokens":out.stat().st_size//4}
        require(out.stat().st_size%4==0 and got==info[mode],f"Token IDs differ from frozen llama.cpp for {mode}")
        if mode=="ordinary":require("roundtrip=PASS" in message,"Missing detokenization round trip")
        results[mode]=got
    return results

def summarize_chat(output,diagnostic,max_new):
    stdout=output.replace(b"\r\n",b"\n")
    parts=stdout.split(USER_PROMPT)
    require(len(parts)==6 and len(stdout.split(ASSISTANT_PROMPT))==4 and
            b"Conversation reset." in parts[3] and not parts[5].strip(),"Missing/extra chat turns or reset")
    events=diagnostic.decode("utf-8",errors="strict").splitlines()
    require(len(events)==4 and events[2]=="[chat-reset]","Unexpected native diagnostics")
    replies=[]
    for fragment,event,turn in zip((parts[1],parts[2],parts[4]),(events[0],events[1],events[3]),(1,2,1)):
        match=re.fullmatch(r"\[chat-turn\] turn=(\d+) finish=(eot|reply-limit) tokens=(\d+) context=(\d+)",event)
        require(match is not None and int(match[1])==turn and 0<int(match[3])<=max_new,"Incomplete accepted turn")
        require(fragment.startswith(ASSISTANT_PROMPT),"Missing assistant output")
        suffix=b"\n" if match[2]=="eot" else b"\n[Chat] reply-limit reached.\n"
        require(fragment.endswith(suffix),"Bad termination evidence")
        raw=fragment[len(ASSISTANT_PROMPT):-len(suffix)]
        answer=raw.decode("utf-8",errors="strict")
        require(any(ch.isprintable() and not ch.isspace() and ch!="\ufffd" for ch in answer),"Empty or invalid visible reply")
        require(all(ch in "\n\t" or ord(ch)>=32 and not 127<=ord(ch)<=159 for ch in answer),"Raw terminal control in reply")
        replies.append({"turn":turn,"finish":match[2],"generated_tokens":int(match[3]),"context":int(match[4]),
                        "text":answer,"utf8_sha256":hashlib.sha256(raw).hexdigest()})
    require(replies[0]==replies[2] and replies[1]["context"]>replies[0]["context"],"History growth/reset replay mismatch")
    return replies

def chat_check(folder,cli,model,protocol):
    folder.mkdir(parents=True)
    transcript="\n".join([protocol["first_user"],protocol["second_user"],":reset",protocol["first_user"],":exit",""])
    (folder/"stdin.txt").write_bytes(transcript.encode("utf-8"))
    env=dict(os.environ,MICIUS_CACHE_ROOT=str(folder/"cache"))
    command=[str(cli),"--chat",str(model),"--context",str(protocol["context"]),
             "--max-new",str(protocol["max_new"]),"--trace-turns"]
    with (folder/"stdin.txt").open("rb") as inp, (folder/"stdout.txt").open("wb") as out, (folder/"stderr.txt").open("wb") as err:
        child=subprocess.Popen(command,stdin=inp,stdout=out,stderr=err,cwd=folder,env=env)
        start=time.monotonic()
        try:
            while True:
                try: code=child.wait(timeout=30);break
                except subprocess.TimeoutExpired: print(f"  {folder.name} 正在运行，已等待 {time.monotonic()-start:.0f} 秒",flush=True)
        except BaseException:
            if child.poll() is None:child.kill()
            child.wait();raise
    save(folder/"exit.json",{"returncode":code})
    require(code==0,"Chat process failed; see "+str(folder/"stderr.txt"))
    replies=summarize_chat((folder/"stdout.txt").read_bytes(),(folder/"stderr.txt").read_bytes(),protocol["max_new"])
    require(all(x["context"]<=protocol["context"] for x in replies),"Invalid context size")
    save(folder/"turns.json",replies)
    return replies

def archive(work):
    target=ROOT/"exports"/(work.name+"-REVIEW.zip")
    target.parent.mkdir(exist_ok=True)
    paths=[work/n for n in ("LAUNCHED","prepared.json","status.json","result.json")]
    paths.extend(sorted((work/"logs").glob("*.log")))
    paths.extend(sorted(p for p in (work/"observations").rglob("*") if p.is_file()))
    part=target.with_suffix(".part")
    with zipfile.ZipFile(part,"x",compression=zipfile.ZIP_DEFLATED) as z:
        for p in paths:z.write(p,p.relative_to(work).as_posix())
    with zipfile.ZipFile(part) as z:
        require(z.testzip() is None and sorted(z.namelist())==sorted(p.relative_to(work).as_posix() for p in paths),"Incomplete archive")
        for p in paths:require(hashlib.sha256(z.read(p.relative_to(work).as_posix())).hexdigest()==digest(p),"Archive data differs")
    part.rename(target)
    print("ARCHIVED",target,flush=True)

def qualify(work,models):
    bins=validate(work)
    state=read(work/"status.json")
    require(state["status"]=="PREPARED","Expected a fresh prepared session")
    with (work/"LAUNCHED").open("x",encoding="utf-8") as f:f.write(str(uuid.uuid4())+"\n")
    try:
        save(work/"status.json",{**state,"status":"RUNNING"})
        protocol=read(ROOT/"validation/m2/protocol.json")
        corpus=read(ROOT/"validation/m2/corpus.json")["text"]
        require(hashlib.sha256(corpus.encode()).hexdigest()==protocol["corpus_utf8_sha256"],"Changed tokenizer fixture")
        paths={k:model_file(models,m) for k,m in protocol["models"].items()}
        checks={}
        for b in PROFILES:
            for k,model in paths.items():
                label=b+"-"+k
                print("词表与模板",label,"（只读元数据）",flush=True)
                info=protocol["models"][k];folder=work/"observations"/label
                checks[label]={"model_sha256":info["model_sha256"],"vocabulary":vocab_check(folder/"tokens",bins[b],model,corpus,protocol,info)}
        for j,(label,case) in enumerate(checks.items(),1):
            b,k=label.split("-");print(f"CHAT {j}/4 {label} 两轮文字、reset 后重放",flush=True)
            case["turns"]=chat_check(work/"observations"/label/"chat",bins[b]["cli"],paths[k],protocol)
        result={"status":"TEXT_CHAT_FUNCTIONAL_PASS","scope":"Fixed UTF-8 dialogs, history and reset replay",
                "prepared_sha256":state["prepared_sha256"],"protocol_sha256":digest(ROOT/"validation/m2/protocol.json"),"checks":checks}
        save(work/"result.json",result);save(work/"status.json",{**state,"status":result["status"]})
        archive(work);print(result["status"],work,flush=True)
    except BaseException as e:
        save(work/"status.json",{**state,"status":"STOPPED","error":str(e),"type":type(e).__name__})
        print("证据保留于",work,file=sys.stderr);raise

def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest="action",required=True)
    sub.add_parser("all");sub.add_parser("prepare")
    sub.add_parser("run").add_argument("directory",type=Path)
    chat=sub.add_parser("chat");chat.add_argument("model",choices=("qwen","smol"))
    chat.add_argument("backend",nargs="?",default="INT8",choices=tuple(PROFILES))
    args=parser.parse_args()
    models=Path(os.environ.get("MICIUS_MODELS",str(Path.cwd()/"models"))).resolve()
    if args.action=="chat":
        pointer=RESULTS/"M2-last-prepared.json"
        if pointer.is_file():
            ref=read(pointer)
            require(re.fullmatch(r"m2-\d{8}-\d{6}-[0-9a-f]{8}",ref["name"]),"Invalid prepared reference")
            work=RESULTS/ref["name"]
            require(digest(work/"prepared.json")==ref["prepared_sha256"],"Changed prepared reference")
        else: work=build()
        bins=validate(work);model=model_file(models,read(ROOT/"validation/m2/protocol.json")["models"][args.model])
        print("聊天模式。输入 :reset 开始新对话，:exit 退出。",flush=True)
        # Inherit the terminal directly. In particular, do not capture its UTF-8 input.
        with subprocess.Popen([str(bins[args.backend]["cli"]),"--chat",str(model)]) as child:
            while True:
                try: return child.wait()
                except KeyboardInterrupt:
                    # The foreground native process receives the console signal
                    # too and discards the interrupted turn. Keep its terminal.
                    continue
    work=args.directory.resolve() if args.action=="run" else build()
    if args.action!="prepare":qualify(work,models)
    return 0
if __name__=="__main__":
    try:sys.exit(main())
    except KeyboardInterrupt:sys.exit("M2 interrupted")
    except Exception as e:sys.exit(f"M2 stopped: {type(e).__name__}: {e}")
