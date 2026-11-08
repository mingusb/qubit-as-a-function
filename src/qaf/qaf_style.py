"""QAF standard formatter. Its output is checked against the parsed program."""
import argparse
import re
from pathlib import Path
from .interpreter import Syntax, parser
from .qaf_layout import expand_layout


TOKENS=re.compile(r'#[^\r\n]*|"[01]+"|[A-Za-z_]\w*|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
                  r'|->|>>|==|!=|<=|>=|\r?\n|[^\s]')
OPS={"+","-","*","/","=","==","!=","<=",">=","<",">","->",">>","|"}
NAMES=re.compile(r"[a-z][a-z_0-9]*\Z")


def semantic(source):
    return parser().parse(expand_layout(source)[0],semantics=Syntax())


def format_source(source):
    expanded,_=expand_layout(source)
    initial=parser().parse(expanded,semantics=Syntax())
    output=[]
    line=""
    depth=group=0
    prev=""
    unary=False
    series=bool(initial.get("model")=="hilbert")
    def flush():
        nonlocal line,prev,unary
        if line.strip():
            output.append("    "*depth+line.rstrip())
        line=""; prev=""; unary=False
    for token in TOKENS.findall(expanded):
        if token in ("\n","\r\n"):
            if group==0 and prev not in OPS|{",","."}: flush()
            continue
        if token.startswith("#"):
            if line: line+="  "
            line+=token
            flush(); continue
        if token=="{":
            flush(); depth+=1; continue
        if token=="}":
            flush(); depth-=1; continue
        if token==";":
            flush(); continue
        if token in {"class","def","proc","qubits","hilbert","qubo","ising"} and not line and depth==0:
            if output and output[-1] and not output[-1].lstrip().startswith("#"):
                output.append("")
        # Rationals in the exact-series grammar are single lexical tokens.
        if token=="/" and series:
            line=line.rstrip()+token; unary=True; prev=token; continue
        if token in ("-","+") and (not prev or prev in OPS|{"(",",","[","return","imag"}):
            if line and prev not in ("(","["): line=line.rstrip()+" "
            line+=token; unary=True; prev=token; continue
        if token in OPS:
            line=line.rstrip()+" "+token+" "
        elif token==",":
            line=line.rstrip()+", "
        elif token==":":
            line=line.rstrip()+": "
        elif token in (")","]","."):
            line=line.rstrip()+token
        elif token in ("(","["):
            space=prev in {"if","while","on","by","return"} or (
                token=="[" and line.startswith("show response "))
            line=(line.rstrip()+" " if space else line)+token
        else:
            if line and not line.endswith((" ","(","[",".")) and not unary:
                line+=" "
            line+=token
        group+=(token in ("(","["))-(token in (")","]"))
        prev=token
        unary=False
    flush()
    result="\n".join(output).strip()+"\n"
    if semantic(result)!=initial:
        raise ValueError("formatter could not preserve this source")
    return result


def names(source):
    for kind,name in re.findall(r"(?m)^\s*(def|proc|fn|let|set|gate|channel|super)\s+([A-Za-z_]\w*)",source):
        if not NAMES.fullmatch(name): yield f"{kind} {name} needs snake_case"


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("files",nargs="+",type=Path)
    cli.add_argument("--write",action="store_true")
    args=cli.parse_args(); errors=[]
    for path in args.files:
        raw=path.read_text(encoding="utf8")
        fixed=format_source(raw)
        if args.write: path.write_text(fixed,encoding="utf8")
        elif raw!=fixed: errors.append(f"{path}: formatting differs")
        errors.extend(f"{path}: {message}" for message in names(fixed))
    if errors: cli.exit(1,"\n".join(errors)+"\n")
