"""Pygments lexers for QAF source and its TatSu grammar."""
from pygments.lexer import RegexLexer,words
from pygments.token import Comment,Keyword,Name,Number,Operator,Punctuation,String,Text

class QAFLexer(RegexLexer):
    name="Qubit-as-a-Function"
    aliases=["qaf"]
    filenames=["*.qaf"]
    tokens={"root":[(r"\s+",Text.Whitespace),(r"#[^\n]*",Comment.Single),
      (words(("class","def","proc","fn","at","return","let","set","if","else","while",
              "place","do","qubits","bits","state","gate","channel","kraus","super",
              "apply","on","invoke","measure","repeat","show","response","probs","prob",
              "amp","density","qubo","ising","linear","pair","offset","hilbert","basis",
              "diagonal","product","imag","tail","fidelity","svm","network","penalty",
              "mulmod","modulo","by","syntax","prefix","pipe","method","field","returns",
              "args","type","term","input","tanh","relu","sigmoid","sine"),suffix=r"\b"),Keyword),
      (r'"[01]+"',String),(r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?",Number),
      (r"[A-Z][A-Za-z_0-9]*",Name.Class),(r"[a-z_][A-Za-z_0-9]*",Name),
      (r"->|>>|==|!=|<=|>=|[+*/<>=|-]",Operator),(r"[][(){},;.:]",Punctuation)]}

class GrammarLexer(RegexLexer):
    name="QAF TatSu grammar"
    aliases=["qafgrammar"]
    tokens={"root":[(r"\s+",Text.Whitespace),(r"@@\w+",Keyword.Declaration),
      (r"'[^']*'",String),(r"/(?:\\.|[^/])*/",String.Regex),(r"\w+(?=\s*=)",Name.Function),
      (r"\w+",Name),(r"::|[@=:$|+*?%]",Operator),(r".",Punctuation)]}
