"""Build a clean Windows SDK and exercise its installed, relocatable targets."""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import sys
from verify_release import MANIFEST, verify

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {"INT8": ("int8-release-windows",30), "F64": ("f64-reference-windows",23)}

def require(ok, message):
    if not ok:
        raise ValueError(message)

def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()

def save(path, data):
    part = path.with_suffix(path.suffix+".tmp")
    part.write_text(json.dumps(data,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    os.replace(part,path)

def execute(command, cwd, log, expected=0):
    command = list(map(str,command))
    print(f"  {log.stem}",flush=True)
    with log.open("wb") as out:
        child = subprocess.Popen(command,cwd=cwd,stdout=out,stderr=subprocess.STDOUT)
        try:
            while True:
                try:
                    code = child.wait(timeout=30)
                    break
                except subprocess.TimeoutExpired:
                    print(f"  {log.stem} 仍在运行",flush=True)
        except BaseException:
            if child.poll() is None:
                child.kill()
            child.wait()
            raise
    if code != expected:
        tail = log.read_bytes()[-4000:].decode("utf-8",errors="replace")
        raise RuntimeError(f"{log.stem} exited {code}, expected {expected}\n{tail}")
    return log.read_text(encoding="utf-8",errors="replace")

def footprint(exe, dest, work):
    execute(["llvm-strip","--strip-all","-o",dest,exe],work,work/(dest.stem+"-strip.log"))
    sections = subprocess.check_output(["llvm-size","-A",dest],text=True)
    docs = subprocess.check_output(["llvm-objdump","-p",dest],text=True,errors="replace")
    match = re.search(r"(?m)^\.text\s+(\d+)",sections)
    require(match,"Missing PE .text section")
    return {"stripped_pe_bytes":dest.stat().st_size,"text_bytes":int(match[1]),
            "imported_dlls":re.findall(r"DLL Name:\s*(\S+)",docs,re.I)}

def prepare(work):
    require(sys.platform=="win32","This functional delivery is qualified for Windows x64 Clang64.")
    for tool in ("clang","clang++","cmake","ctest","ninja","llvm-strip","llvm-size","llvm-objdump"):
        require(shutil.which(tool),f"Missing tool: {tool}")
    schema = subprocess.check_output(["clang++","-dumpmachine"],text=True).strip()
    require(schema in ("x86_64-w64-windows-gnu","x86_64-w64-mingw32"),"Expected Windows x64 Clang64")
    verify()
    sys.path.insert(0,str(ROOT/"tests"))
    from reference.make_fixture import make_fixture
    logs=work/"logs"
    logs.mkdir()
    scratch=work/"tmp"
    scratch.mkdir()
    os.environ["TMP"]=os.environ["TEMP"]=str(scratch)
    package_hash=digest(MANIFEST)
    protocol=json.loads((ROOT/"validation/m1/protocol.json").read_text(encoding="utf-8"))
    old_header=ROOT/"tests/reference/v07/micius_runtime.h"
    canonical_header=old_header.read_bytes().replace(b"\r\n",b"\n")
    require(hashlib.sha256(canonical_header).hexdigest()==protocol["v07_header_sha256"],"v0.7 header identity mismatch")
    require((ROOT/"include/micius_runtime.h").read_bytes().replace(b"\r\n",b"\n")==canonical_header,"Runtime V1 header changed")
    old_object=work/"v07-client.obj"
    execute(["clang","-std=c11","-I",old_header.parent,"-c",ROOT/"tests/abi_v07_client.c","-o",old_object],
            work,logs/"compile-v07-client.log")
    old_sha=digest(old_object)
    external=work/"external-source"
    shutil.copytree(ROOT/"examples/file-consumer",external)
    fixtures={k:make_fixture(work/("tiny-"+k+".gguf"),architecture=k,context=16) for k in ("llama","qwen2")}
    inputs=work/"tiny-ids.u32"
    inputs.write_bytes(struct.pack("<4I",3,61,7,11))
    sdk=work/"sdk"
    sdk.mkdir()
    results={}
    for backend,(preset,count) in PROFILES.items():
        print(f"{backend}：构建、CTest、安装和外部 C 调用",flush=True)
        build=work/("build-"+backend)
        stage=work/("install-"+backend)
        prefix=sdk/backend
        execute(["cmake","--preset",preset,"-B",build],ROOT,logs/(backend+"-configure.log"))
        execute(["cmake","--build",build,"--parallel","6"],work,logs/(backend+"-build.log"))
        tests=execute(["ctest","--test-dir",build,"--output-on-failure"],work,logs/(backend+"-ctest.log"))
        require(re.search(rf"100% tests passed(?:, 0 tests failed)? out of {count}\b",tests),f"Unexpected test count: {backend}")
        execute(["cmake","--install",build,"--prefix",stage],work,logs/(backend+"-install.log"))
        require(stage.resolve().parent==work and prefix.resolve().parent==sdk, "Unexpected SDK path")
        stage.rename(prefix)
        require((prefix/"include/micius_gguf.h").is_file() and (prefix/"include/micius_runtime.h").is_file(),"Incomplete SDK")
        for file in (prefix/"lib/cmake/MICIUS").glob("*.cmake"):
            exported=file.read_text(encoding="utf-8").replace("\\","/").lower()
            for private in (stage,build,ROOT/"src",ROOT/"include"):
                require(private.as_posix().lower() not in exported,"Install export contains private build paths")
        modes={}
        outputs={}
        for mode in ("llama","qwen2","both"):
            label=backend+"-"+mode
            project=work/("consumer-"+label)
            args=["cmake","-S",external,"-B",project,"-G","Ninja","-DCMAKE_BUILD_TYPE=Release",
                  "-DCMAKE_C_COMPILER=clang","-DCMAKE_CXX_COMPILER=clang++",
                  "-DMICIUS_DIR="+str(prefix/"lib/cmake/MICIUS"),"-DPARSERS="+mode]
            if mode=="both":
                args.append("-DV07_CLIENT_OBJECT="+str(old_object))
            execute(args,work,logs/(label+"-configure.log"))
            link=execute(["cmake","--build",project,"--verbose"],work,logs/(label+"-link.log"))
            exe=project/"micius_model_probe.exe"
            stripped=project/("stripped-"+label+".exe")
            size=footprint(exe,stripped,logs)
            result={"executable":exe.relative_to(work).as_posix(),"sha256":digest(exe),"footprint":size,"fixtures":{}}
            for arch,fixture in fixtures.items():
                selected=mode=="both" or mode==arch
                require(("libmicius_parser_"+arch+".a" in link)==selected,"Unexpected parser archive on link line")
                target=project/(arch+".f32")
                msg=execute([stripped,fixture,work/"test-cache",inputs,target],work,
                            logs/(label+"-"+arch+".log"),0 if selected else 1)
                if selected:
                    require("reset_replay=PASS" in msg and f"architecture={arch}" in msg and target.stat().st_size==4*4*128,
                            "Incomplete external decode")
                    h=digest(target)
                    require(h==outputs.setdefault(arch,h),"Different output from the two linkage combinations")
                    result["fixtures"][arch]=h
                else:
                    require("No linked parser for architecture: "+arch in msg and not target.exists(),
                            "Missing parser was not diagnosed")
                    result["fixtures"][arch]="UNLINKED_REJECTED"
            if mode=="both":
                execute([project/"v07_client.exe"],work,logs/(backend+"-v07-relink.log"))
                require(digest(old_object)==old_sha,"v0.7 client object changed")
            modes[mode]=result
        results[backend]={"ctest_passed":count,"modes":modes}
    return {"status":"PREPARED","manifest_sha256":package_hash,"v07_client_object_sha256":old_sha,
            "compiler":subprocess.check_output(["clang++","--version"],text=True).splitlines()[0],
            "profiles":results,"real_model_runs":0}
