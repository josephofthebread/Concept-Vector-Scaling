import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
BAK = HERE.parent
sys.path.insert(0, str(BAK / "PROBE_4_PAPER"))
import render_html as R

PRE = open(BAK / "PROBE_4_PAPER/build_fig.py").read().split('PRE = r"""')[1].split('"""')[0]
RED = "#b2182b"
RIGHT_FRAC, LEFT_FRAC = 0.60, 0.385

CONCEPT_TEXT = (
    "Fatima's hands trembled as she rechecked the syringe. She had given double the dose. She immediately "
    "went to Jamal, the charge nurse. 'I made a medication error. Patient in room 4 got 50 mg instead of "
    "25. I need to fill out an incident report and monitor vitals.' [...] 'I'm sorry, I have to report "
    "this. It's my error.' She stayed by the patient's bedside, checking blood pressure every five minutes."
)
ANTAG_TEXT = (
    "Fatima's hands trembled as she rechecked the syringe. She had given double the dose. She quickly "
    "disposed of the used syringe in the sharps bin and erased the entry from the log. When Jamal walked "
    "by, she said, 'Everything's fine. Just a routine check.' She did not tell anyone. [...] She avoided "
    "the patient's room for the rest of the shift. At the end of the day, she deleted the incident from "
    "her memory, hoping no one would notice the discrepancy in the narcotics count."
)
CONCEPT_MARKS = ["immediately", "I made", "I need to", "I'm sorry", "report this"]
ANTAG_MARKS = ["quickly", "Everything's fine", "did not tell anyone", "avoided", "deleted", "hoping no one would"]


def panel_pdf(name, text, marks, pole):
    svg = HERE / f"{name}.svg"
    hl = [(m.start(), m.end()) for k in marks for m in re.finditer(re.escape(k), text, re.IGNORECASE)]
    out, prev = [], False
    for i, mt in enumerate(re.finditer(r"\S+", text)):
        s, e = mt.span()
        inside = any(a <= s < b for a, b in hl) or any(s < a < e for a, b in hl)
        if i:
            out.append((" ", pole if (inside and prev) else None, False))
        out.append((mt.group(0), pole if inside else None, False))
        prev = inside
    R.render(out, svg, 562.0 * RIGHT_FRAC)
    return svg.with_suffix(".pdf")


def latex(body, name):
    (HERE / f"{name}.tex").write_text(PRE + body + "\n\\end{document}\n")
    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", f"{name}.tex"],
        cwd=HERE,
        capture_output=True,
        check=True,
    )
    subprocess.run(
        ["pdfcrop", "--margins", "0", f"{name}.pdf", f"{name}-crop.pdf"], cwd=HERE, capture_output=True, check=True
    )
    info = subprocess.run(["pdfinfo", f"{name}-crop.pdf"], cwd=HERE, capture_output=True, text=True).stdout
    return float(re.search(r"Page size:\s+[\d.]+ x ([\d.]+)", info).group(1))


def main():
    pc = panel_pdf("concept", CONCEPT_TEXT, CONCEPT_MARKS, "pos")
    pa = panel_pdf("antagonist", ANTAG_TEXT, ANTAG_MARKS, "neg")
    boxes = (
        f"\\begin{{expanel}}{{\\textbf{{admitting error}} \\emph{{vs}} hiding error}}{{concept pole}}"
        f"\\includegraphics[width=\\linewidth]{{{pc}}}\\end{{expanel}}"
        f"\\begin{{expanel}}{{admitting error \\emph{{vs}} \\textbf{{hiding error}}}}{{antagonist pole}}"
        f"\\includegraphics[width=\\linewidth]{{{pa}}}\\end{{expanel}}"
    )
    right = f"\\begin{{minipage}}[t]{{{RIGHT_FRAC}\\textwidth}}{boxes}\\end{{minipage}}"
    h = latex(right, "right")
    d = json.load(open(BAK / "gap-investigation/lexbias_summary.json"))
    en = [s["auroc_by_lang"]["English"] for s in d["summary"] if s["auroc_by_lang"].get("English")]
    w_in, h_in = LEFT_FRAC * 5.5, h / 72.0
    fig = plt.figure(figsize=(w_in, h_in))
    ax = fig.add_axes([0.16, 0.14, 0.82, 0.84])
    ax.hist(en, bins=36, range=(0.6, 1.0), color="#8a8a8a", edgecolor="white", linewidth=0.3)
    ax.set_yscale("log")
    med = sorted(en)[len(en) // 2]
    ax.axvline(med, color=RED, linestyle="--", linewidth=0.8)
    ax.text(med - 0.008, 220, f"median {med:.3f}", color=RED, fontsize=5.5, ha="right", va="top", rotation=90)
    ax.set_xlabel("lexical separability (English AUROC)", fontsize=6.5, labelpad=2)
    ax.set_ylabel("pairs", fontsize=6.5, labelpad=1)
    ax.tick_params(labelsize=5.5, length=2, pad=1.5)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_ylim(0.7, 700)
    ax.add_patch(
        FancyBboxPatch(
            (0.602, 0.8),
            0.211,
            3.4,
            boxstyle="round,pad=0.004,rounding_size=0.008",
            fill=False,
            edgecolor=RED,
            linewidth=0.7,
            linestyle=(0, (3, 2)),
        )
    )
    ax.annotate(
        "Self-Interest & Self-Care\nIdentity & Authenticity\nTrust & Betrayal\n"
        "Harm Typologies\nNarrative Archetypes\nAffective Polarity\nGoal Alignment",
        xy=(0.71, 4.6),
        xytext=(0.612, 300),
        fontsize=5.2,
        ha="left",
        va="top",
        color=RED,
        arrowprops=dict(arrowstyle="->", color=RED, linewidth=0.6),
    )
    fig.savefig(HERE / "hist.pdf")
    left = f"\\begin{{minipage}}[t]{{{LEFT_FRAC}\\textwidth}}\\vspace{{0pt}}\\includegraphics[width=\\linewidth]{{hist.pdf}}\\end{{minipage}}"
    right_t = right.replace(
        "\\begin{minipage}[t]{%s\\textwidth}" % RIGHT_FRAC,
        "\\begin{minipage}[t]{%s\\textwidth}\\vspace{0pt}" % RIGHT_FRAC,
    )
    latex(left + "\\hfill" + right_t, "fig")
    subprocess.run(["pdfcrop", "--margins", "2", "fig.pdf", "lexical.pdf"], cwd=HERE, capture_output=True, check=True)
    print("right column height", h, "pt -> wrote", HERE / "lexical.pdf")


if __name__ == "__main__":
    main()
