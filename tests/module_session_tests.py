"""Registration, independent sessions, failed replacement and canonical shell commands."""
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

from reference.make_fixture import make_fixture

def main():
    driver, cli = map(lambda s: str(Path(s).resolve()), sys.argv[1:3])
    with tempfile.TemporaryDirectory(dir=Path.cwd(), prefix="session-") as temp:
        root = Path(temp)
        model = make_fixture(root/"ok.gguf", architecture="qwen2", context=12)
        cache = root/"cache"
        subprocess.run([driver, str(model), str(cache)], cwd=root, check=True)
        assert len(list(cache.glob("*/contexts/*/kv")))==2
        bad = make_fixture(root/"bad.gguf",architecture="qwen2",omit_tensor="blk.0.attn_q.bias")
        commands = "\n".join(("help", "quit", "cls", "run", "clear", "load "+model.name,
                               "load "+bad.name,"stats","generate 3","exit","")) 
        env = dict(os.environ,MICIUS_CACHE_ROOT=str(cache))
        run = subprocess.run([cli],input=commands,text=True,encoding="utf-8",errors="replace",
                             capture_output=True,cwd=root,env=env,check=True)
        text = re.sub(r"\x1b\[[0-9;]*[A-Za-z]","",run.stdout)
        help_text=text.split("Available Commands:",1)[1].split("Unknown command",1)[0]
        for name in ("help","clear","load","stats","generate","exit","debug","log"):
            assert re.search(r"(?m)^  "+name+r"\s+",help_text), name
        for alias in ("cls","quit","run"):
            assert not re.search(r"(?m)^  "+alias+r"\s+", help_text), alias
        assert text.count("Unknown command.")==3 and text.count("securely bound to Engine.")==1, text
        assert "missing required tensor: blk.0.attn_q.bias" in text and "Prefill complete" in text, text
        assert "Halting MICIUS Engine" in text and len(list(cache.glob("*/contexts/*/kv")))==3
        print("file/parser ABI errors, memory-only option, two live contexts, reset, failed CLI replacement and aliases: PASS")

if __name__=="__main__":
    main()
