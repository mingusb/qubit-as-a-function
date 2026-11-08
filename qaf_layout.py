"""Translate optional indentation blocks to explicit braces with a source-offset map."""
import re


def expand_layout(source):
    """Return TatSu input plus the original offset at every translated boundary."""
    lines = source.splitlines(keepends=True)
    code = [line.partition("#")[0].rstrip() for line in lines]
    following = ""
    next_code = [""] * len(lines)
    for i in range(len(lines)-1, -1, -1):
        next_code[i] = following
        if code[i].strip():
            following = code[i].lstrip()
    out, offsets, blocks = [], [0], []
    start = depth = explicit = 0
    logical = ""

    def insert(s, position):
        out.append(s)
        offsets.extend([position]*len(s))

    def copy(a, b):
        out.append(source[a:b])
        offsets.extend(range(a+1,b+1))

    for i, line in enumerate(lines):
        end = start+len(line)
        useful = code[i].strip()
        if not useful:
            copy(start,end); start=end
            continue
        prefix = code[i][:len(code[i])-len(code[i].lstrip(" \t"))]
        column = len(prefix.expandtabs(8))
        if depth==0:
            logical = useful
            while blocks and blocks[-1]["explicit"]==explicit:
                frame=blocks[-1]
                if frame["body"] is None:
                    if column<=frame["header"]:
                        raise ValueError(f"line {i+1}: expected an indented body")
                    frame["body"]=(column,prefix)
                    break
                if column<=frame["header"]:
                    insert("} ",start); blocks.pop()
                    continue
                if (column,prefix)!=frame["body"]:
                    raise ValueError(f"line {i+1}: align this statement with its block")
                break
        else:
            logical += " "+useful
        depth += sum(useful.count(c) for c in "([")-sum(useful.count(c) for c in ")]")
        header = (re.match(r"^(?:class|def|proc|if|else|while|repeat|hilbert|network)\b",logical)
                  or re.search(r"\bkraus\s*:?$",logical))
        implicit = (depth==0 and header and "{" not in logical and "}" not in logical
                    and not next_code[i].startswith("{"))
        if implicit:
            last=start+len(code[i])
            colon=source[last-1]==":"
            copy(start,last-int(colon))
            if colon:
                out.append("{"); offsets.append(last)
            else:
                insert(" {",last)
            copy(last,end)
            blocks.append({"header":column,"body":None,"explicit":explicit})
        else:
            copy(start,end)
            explicit+=useful.count("{")-useful.count("}")
        start=end
    if any(b["body"] is None for b in blocks):
        raise ValueError("expected an indented body at end of file")
    if blocks:
        # A final source comment may have no terminating newline.
        insert("\n"+"} "*len(blocks),len(source))
    return "".join(out),offsets
