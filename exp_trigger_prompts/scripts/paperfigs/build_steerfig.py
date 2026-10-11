import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BAK = HERE.parent
PAPER = BAK.parent / "report_ICLR2027"
sys.path.insert(0, str(BAK / "REPORT_ABL_3"))
os.chdir(BAK / "REPORT_ABL_3")
import make_steerfig as M

LETTERS = "ABC"


def tex_escape(s):
    s = s.replace("\\", r"\textbackslash{}")
    for a, b in (
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ):
        s = s.replace(a, b)
    return s


def body(text, markers=None):
    """Words -> TeX; runs of marker words become one \\textbf{...}; [...] -> \\ldots."""
    out, run = [], []
    for w in text.split():
        if w == "[...]":
            if run:
                out.append(r"\textbf{" + " ".join(run) + "}")
                run = []
            out.append(r"\ldots")
            continue
        if markers and M.is_marker(w, markers):
            run.append(tex_escape(w))
            continue
        if run:
            out.append(r"\textbf{" + " ".join(run) + "}")
            run = []
        out.append(tex_escape(w))
    if run:
        out.append(r"\textbf{" + " ".join(run) + "}")
    s = " ".join(out)
    s = re.sub(r'(^|\s|\{)"', r"\1``", s).replace('"', "''")
    return s.replace(r"\ldots ", r"\ldots\ ")


def main():
    mk = {r["pair_id"]: set(r["markers"]) for r in json.load(open("../port/markerinject.json"))}
    cols = []
    for (tag, model, concept, pid, alpha, layer, slot), L in zip(M.SEL, LETTERS):
        rows = [json.loads(l) for l in open(f"../port/steer-examples-{tag}.jsonl")]
        st = [r for r in rows if (r["pair_id"], r["alpha"], r["layer"], r["slot"]) == (pid, alpha, layer, slot)][0]
        base = [r for r in rows if r["pair_id"] == pid and r["alpha"] == 0 and r["slot"] == slot][0]
        assert base["prompt"] == st["prompt"]
        markers = M.marker_words(mk, pid)
        b = M.trim(base["response"].replace("\n", " "), M.TRIM)
        s = M.trim(st["response"].replace("\n", " "), M.TRIM, markers)
        cols.append(
            dict(
                L=L,
                model=model,
                concept=concept,
                alpha=alpha,
                layer=layer,
                prompt=body(st["prompt"]),
                base=body(b),
                steered=body(s, markers),
            )
        )
    w = "0.328"

    def row(kind, key, title_fn):
        cells = []
        for c in cols:
            cells.append(
                f"    \\begin{{minipage}}[t]{{{w}\\linewidth}}\n"
                f"        \\begin{{{kind}}}[equal height group=s29{key}]{{{title_fn(c)}}}\n"
                f"            \\scriptsize\n            {c[key]}\n"
                f"        \\end{{{kind}}}\n    \\end{{minipage}}"
            )
        return "\\hfill\n".join(cells)

    frag = (
        "\\centering\n"
        "    % --- Row 1: prompts (labels live in these titles) ---\n"
        + row("panelbase", "prompt", lambda c: f"({c['L']}) Prompt ({c['model']})")
        + "\\\\[1pt]\n"
        "    % --- Row 2: baselines ---\n"
        + row("panelbase", "base", lambda c: "Baseline --- no steering")
        + "\\\\[1pt]\n"
        "    % --- Row 3: steered ---\n"
        + row("panelpos", "steered", lambda c: f"$+{c['alpha']}$ $\\rightarrow$ {c['concept']}")
        + "\n"
    )
    (HERE / "fragment.tex").write_text(frag)
    pre = (PAPER / "paper.tex").read_text()
    lines = pre.splitlines()
    defs = "\n".join(
        [l for l in lines if l.startswith("\\definecolor")]
        + lines[
            lines.index(next(l for l in lines if l.startswith("\\newtcolorbox{panelprom}"))) : lines.index(
                next(l for l in lines if l.startswith("\\newtcolorbox{panelneg}"))
            )
            + 2
        ]
    )
    doc = (
        "\\documentclass{article}\n\\usepackage[papersize={5.5in,11in},margin=0in]{geometry}\n"
        "\\usepackage{times}\\usepackage{xcolor}\\usepackage{tcolorbox}\\tcbuselibrary{skins}\n"
        f"{defs}\n\\pagestyle{{empty}}\\setlength{{\\parindent}}{{0pt}}\n\\begin{{document}}\n"
        f"\\begin{{minipage}}{{\\linewidth}}\n{frag}\\end{{minipage}}\n\\end{{document}}\n"
    )
    (HERE / "preview.tex").write_text(doc)
    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", "preview.tex"],
        cwd=HERE,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["pdfcrop", "--margins", "2", "preview.pdf", "preview-crop.pdf"], cwd=HERE, capture_output=True, check=True
    )
    print("wrote", HERE / "fragment.tex", "and preview-crop.pdf")


if __name__ == "__main__":
    main()
