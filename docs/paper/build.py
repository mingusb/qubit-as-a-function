"""Build the original article from the repository's runnable sources and minted cache."""
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile


def main():
    article = Path(__file__).resolve().parent
    repository = article.parents[1]
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--output", type=Path, default=article / "manuscript.pdf")
    command.add_argument("--refresh", action="store_true", help="Run the installed minted lexer")
    args = command.parse_args()
    with tempfile.TemporaryDirectory(prefix="qaf-paper-") as temporary:
        build = Path(temporary)
        for pattern in ("*.tex", "*.cls", "responses.pdf"):
            for source in article.glob(pattern):
                shutil.copyfile(source, build / source.name)
        shutil.copytree(article / "_minted", build / "_minted")
        shutil.copytree(repository / "examples", build / "examples", ignore=shutil.ignore_patterns("__pycache__"))
        language = repository / "src/qaf"
        shutil.copytree(language / "lean", build / "lean", ignore=shutil.ignore_patterns("*.olean"))
        for grammar in language.glob("*.ebnf"):
            shutil.copyfile(grammar, build / grammar.name)
        flags = ["-shell-escape" if args.refresh else "-no-shell-escape", "-halt-on-error", "-interaction=nonstopmode"]
        for _ in range(2):
            result = subprocess.run(["pdflatex", *flags, "paper.tex"], cwd=build, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if result.returncode:
                raise SystemExit(result.stdout.decode(errors="replace")[-5000:])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(build / "paper.pdf", args.output)
        print(args.output.resolve())


if __name__ == "__main__":
    main()
