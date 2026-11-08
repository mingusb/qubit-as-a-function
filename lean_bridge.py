"""Transpile to and from closed Lean QAF.Term values; Lean implements both directions."""
import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from tatsu.grammars import ModelContext
from qaf import parser


@dataclass
class Span:
    rule:str
    start:int
    end:int
    children:list


class Concrete(ModelContext):
    def _get_node(self,pos,ruleinfo):
        def children(v):
            if isinstance(v,Span): yield v
            elif isinstance(v,(list,tuple)):
                for item in v: yield from children(item)
        return Span(ruleinfo.name,pos,self._pos,list(children(self.cst)))


def concrete(source):
    from qaf_layout import expand_layout
    from qaf_prefix import is_prefix,layout,grammar as prefix_grammar
    if is_prefix(source):
        expanded,offsets=layout(source)
        grammar=prefix_grammar()
    else:
        expanded,offsets=expand_layout(source)
        grammar=parser()
    ctx=Concrete(grammar.rules,config=grammar.config)
    root=grammar.parse(expanded,ctx=ctx)
    def folded(sp):
        items=[]
        start,end=offsets[sp.start],offsets[sp.end]
        cursor=start
        for child in sp.children:
            a,b=offsets[child.start],offsets[child.end]
            if not cursor<=a<=b<=end:
                raise ValueError("invalid concrete syntax span")
            if a>cursor: items.append({"token":source[cursor:a]})
            items.append(folded(child))
            cursor=b
        if cursor<end: items.append({"token":source[cursor:end]})
        acc="empty"
        for item in reversed(items): acc={"cons":[item,acc]}
        return {"rule":[sp.rule,acc]}
    root.start,root.end=0,len(expanded)
    return folded(root)


def spell(tree):
    if tree=="empty": return ""
    if "token" in tree: return tree["token"]
    if "cons" in tree: return "".join(spell(a) for a in tree["cons"])
    return spell(tree["rule"][1])


def convert(mode,source,dest):
    lean=os.getenv("LEAN") or shutil.which("lean")
    if not lean: raise RuntimeError("put Lean 4.24.0 on PATH or set LEAN")
    root=Path(__file__).resolve().parent/"lean"
    with tempfile.TemporaryDirectory() as folder:
        tmp=Path(folder)
        shutil.copy2(root/"QAF.lean",tmp/"QAF.lean")
        env={**os.environ,"LEAN_PATH":str(tmp)}
        def command(args,data=None):
            p=subprocess.run([lean,*args],cwd=tmp,env=env,input=data,capture_output=True)
            if p.returncode: raise RuntimeError((p.stdout+p.stderr).decode(errors="replace")[-2500:])
            return p.stdout
        command(["-o","QAF.olean","QAF.lean"])
        def read_value(file):
            return json.loads(command(["--run",str(file)]))
        if mode=="to-lean":
            original=source.read_bytes().decode("utf-8")
            tree=concrete(original)
            if spell(tree)!=original: raise ValueError("source witness failed")
            text=command(["--run",str(root/"Emit.lean")],json.dumps(tree).encode())
            generated=tmp/"Program.lean"
            generated.write_bytes(text)
            if read_value(generated)!=tree: raise ValueError("Lean export mismatch")
        else:
            tree=read_value(source.resolve())
            original=spell(tree)
            if concrete(original)!=tree: raise ValueError("term is outside the grammar's canonical image")
            text=original.encode("utf-8")
        dest.write_bytes(text)


if __name__=="__main__":
    arg=argparse.ArgumentParser(description=__doc__)
    arg.add_argument("mode",choices=["to-lean","from-lean"])
    arg.add_argument("source",type=Path)
    arg.add_argument("dest",type=Path)
    a=arg.parse_args()
    convert(a.mode,a.source,a.dest)
