"""M1 user-terminal functional checks. No timing or performance classification."""
import argparse
import array
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
import uuid
import zipfile
import zlib

from m1_prepare import PROFILES, ROOT, digest, execute, prepare, require, save
from verify_release import MANIFEST, verify

def read(path):
    return json.loads(path.read_text(encoding="utf-8"))

def check_rows(actual, reference, vocab, ids, gates):
    if sys.byteorder != "little":
        raise ValueError("Expected little-endian float32 rows")
    n = len(ids)-1
    require(n > 0 and len(actual)==len(reference)==4*vocab*n,"Unexpected logits length")
    scores=[]
    def target_nll(row,target):
        peak=max(row)
        return peak+math.log(sum(math.exp(x-peak) for x in row))-row[target]
    for pos in range(n):
        start=pos*vocab*4
        end=start+vocab*4
        a=array.array("f"); a.frombytes(actual[start:end])
        r=array.array("f"); r.frombytes(reference[start:end])
        require(all(map(math.isfinite,a)) and all(map(math.isfinite,r)),"Non-finite logits")
        mean_r=sum(r)/vocab
        shift=sum(a)/vocab-mean_r
        norm=sum((x-mean_r)**2 for x in r)
        require(norm>0,"Degenerate reference logits")
        distance=math.sqrt(sum((x-y-shift)**2 for x,y in zip(a,r))/norm)
        delta=target_nll(a,ids[pos+1])-target_nll(r,ids[pos+1])
        scores.append({"position":pos,"centered_relative_l2":distance,"target_delta_nll":delta})
    worst=max(x["centered_relative_l2"] for x in scores)
    mean=sum(abs(x["target_delta_nll"]) for x in scores)/n
    return {"rows":scores,"max_centered_relative_l2":worst,"mean_absolute_target_delta_nll":mean,
            "passed":worst<=gates["max_centered_relative_l2_per_position"] and
                     mean<=gates["mean_absolute_target_nll_difference_nats"]}

def sample(work, models):
    prep=read(work/"prepared.json")
    require(prep.get("status")=="PREPARED","Preparation incomplete")
    require(digest(work/"prepared.json")==read(work/"status.json")["prepared_sha256"],
            "Prepared receipt changed")
    require(digest(MANIFEST)==prep["manifest_sha256"],"Package changed since preparation")
    verify()
    protocol=read(ROOT/"validation/m1/protocol.json")
    out=work/"observations"
    out.mkdir()
    qref=zlib.decompress((ROOT/"validation/m1/qwen-s00-prefix8.f32.zlib").read_bytes())
    require(hashlib.sha256(qref).hexdigest()==protocol["models"]["qwen"]["reference"]["LLAMA"]["prefix_sha256"],
            "Reference slice hash mismatch")
    for info in protocol["models"].values():
        model=models/info["filename"]
        print("核对实机模型身份："+model.name,flush=True)
        require(model.is_file() and model.stat().st_size==info["bytes"] and digest(model)==info["model_sha256"],
                "Model identity mismatch: "+str(model))
    entries=[]
    for backend in PROFILES:
        tool=prep["profiles"][backend]["modes"]["both"]
        exe=work/tool["executable"]
        require(digest(exe)==tool["sha256"],f"Changed installed consumer: {backend}")
        for k,model in protocol["models"].items():
            print(f"功能检查 {len(entries)+1}/4：{backend} / {model['architecture']}，8 个固定位置和 reset 重放",flush=True)
            case=out/(backend+"-"+k)
            case.mkdir()
            ids=model["ids_with_next_target"]
            require(len(ids)==protocol["steps"]+1 and all(0<=x<model["vocab"] for x in ids),"Invalid fixed IDs")
            inp=case/"tokens.u32"
            inp.write_bytes(struct.pack("<"+"I"*(len(ids)-1),*ids[:-1]))
            output=case/"logits.f32"
            text=execute([exe,models/model["filename"],work/"context-cache",inp,output],case,case/"native.log")
            fields=dict(line.split("=",1) for line in text.splitlines() if "=" in line)
            require(fields.get("reset_replay")=="PASS" and fields.get("steps")==str(protocol["steps"])
                    and fields.get("architecture")==model["architecture"],"Incomplete functional evidence")
            directory=Path(fields["context_directory"]).resolve()
            parent=(work/"context-cache"/model["model_sha256"]/"contexts").resolve()
            require(directory.parent==parent and re.fullmatch(r"[0-9a-f]{32}",directory.name)
                    and (directory/"kv").is_dir() and all(e["directory"]!=str(directory) for e in entries),
                    "Context directory identity mismatch")
            h=digest(output)
            require(output.stat().st_size==protocol["steps"]*model["vocab"]*4,"Incomplete logits output")
            if k=="smol":
                label="I1" if backend=="INT8" else "F64"
                metrics={"reference":label,"passed":h==model["reference"][label]["prefix_sha256"]}
            else:
                metrics=check_rows(output.read_bytes(),qref,model["vocab"],ids,protocol["qwen_gates"])
            entry={"case":case.name,"model_sha256":model["model_sha256"],"probe_sha256":tool["sha256"],
                   "tokens_sha256":digest(inp),"logits_sha256":h,"directory":str(directory),
                   "steps":protocol["steps"],"reset_replay":True,"comparison":metrics}
            save(case/"check.json",entry)
            entries.append(entry)
    ok=all(e["comparison"]["passed"] for e in entries)
    return {"status":"FUNCTIONAL_CHECK_PASSED" if ok else "NUMERICAL_GATE_NOT_MET",
            "scope":"Two fixed GGUF files, s00 prefix, single-thread functional check",
            "protocol_sha256":digest(ROOT/"validation/m1/protocol.json"),"prepared_sha256":digest(work/"prepared.json"),
            "checks":entries}

def pack(work):
    exports=ROOT/"exports"
    exports.mkdir(exist_ok=True)
    target=exports/(work.name+"-REVIEW.zip")
    pending=target.with_suffix(".part")
    paths=[work/n for n in ("prepared.json","LAUNCHED","status.json","result.json")]
    paths+=sorted((work/"observations").glob("*/*"))
    paths+=sorted((work/"logs").glob("*.log"))
    with zipfile.ZipFile(pending,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in paths:
            z.write(p,p.relative_to(work).as_posix())
    with zipfile.ZipFile(pending) as z:
        require(z.testzip() is None and len(z.namelist())==len(paths),"Archive integrity check failed")
        for p in paths:
            require(hashlib.sha256(z.read(p.relative_to(work).as_posix())).hexdigest()==digest(p),"Archive payload mismatch")
    pending.rename(target)
    print("ARCHIVED",target,flush=True)

def main():
    cli=argparse.ArgumentParser()
    cli.add_argument("action",choices=("all","prepare","run"))
    cli.add_argument("directory",nargs="?",type=Path)
    args=cli.parse_args()
    require((args.action=="run")==bool(args.directory),"Only run accepts an existing prepared directory")
    if args.action=="run":
        work=args.directory.resolve()
        require(work.parent==(ROOT/"results").resolve(),"Expected this package's prepared results directory")
    else:
        results=ROOT/"results"
        results.mkdir(exist_ok=True)
        work=results/("m1-"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+"-"+uuid.uuid4().hex[:8])
        work.mkdir()
        print("WORK",work,flush=True)
        save(work/"status.json",{"status":"PREPARING"})
        try:
            save(work/"prepared.json",prepare(work))
            save(work/"status.json",{"status":"PREPARED","prepared_sha256":digest(work/"prepared.json")})
        except BaseException as err:
            save(work/"status.json",{"status":"PREPARATION_STOPPED","type":type(err).__name__,"error":str(err)})
            raise
        print("PREPARED",work,flush=True)
        if args.action=="prepare": return 0
    status=read(work/"status.json")
    require(status["status"]=="PREPARED","Expected a fresh prepared session")
    # Exclusive launch. A stopped session remains evidence and cannot be spliced into a later run.
    with (work/"LAUNCHED").open("x",encoding="utf-8") as f:
        f.write(str(uuid.uuid4())+"\n")
    status["status"]="RUNNING"
    save(work/"status.json",status)
    try:
        models=Path(os.environ.get("MICIUS_MODELS",str(Path.cwd()/"models"))).resolve()
        result=sample(work,models)
        save(work/"result.json",result)
        save(work/"status.json",{**status,"status":result["status"]})
        pack(work)
        print(result["status"],work,flush=True)
        return 0 if result["status"]=="FUNCTIONAL_CHECK_PASSED" else 1
    except BaseException as err:
        save(work/"status.json",{**status,"status":"STOPPED","type":type(err).__name__,"error":str(err)})
        print("证据保留于",work,file=sys.stderr)
        raise

if __name__=="__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit("M1 interrupted")
    except Exception as err:
        sys.exit(f"M1 stopped: {type(err).__name__}: {err}")
