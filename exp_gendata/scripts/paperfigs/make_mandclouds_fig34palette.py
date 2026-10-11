import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from safetensors import safe_open

RED = "#b2182b"
LANGS = ["English", "German", "Mandarin", "Russian", "Spanish"]
LCOL = {"English": "#2166ac", "German": "#1b7837", "Mandarin": "#762a83", "Russian": "#b2182b", "Spanish": "#e66101"}
PAIRS = [(77, "attention to detail"), (193, "conscientiousness"), (318, "emotional load"), (400, "formality")]


def clouds():
    langids = json.load(open("langids_4pairs.json"))
    fig, axes = plt.subplots(1, 4, figsize=(10.4, 2.6))
    with safe_open("../psanalysis-perstory.safetensors", framework="np") as f:
        for ax, (pid, name) in zip(axes, PAIRS):
            xy = f.get_tensor(f"p{pid}_L25_k1")
            lid = np.array(langids[str(pid)])
            for i, lang in enumerate(LANGS):
                m = lid == i
                ax.scatter(
                    xy[m, 0],
                    xy[m, 1],
                    s=3.5,
                    c=LCOL[lang],
                    label=lang,
                    alpha=0.9 if lang == "Mandarin" else 0.75,
                    linewidths=0,
                )
            ax.set_title(name, fontsize=9.5)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.5)
                sp.set_color("#999999")
    axes[0].legend(
        fontsize=6.6,
        loc="lower left",
        frameon=False,
        handletextpad=0.1,
        borderaxespad=0.15,
        labelspacing=0.25,
        markerscale=1.8,
    )
    fig.subplots_adjust(left=0.008, right=0.995, top=0.88, bottom=0.02, wspace=0.05)
    fig.savefig("../MANDFIG_30/mandarin-clouds.pdf")
    print("wrote fig/mandarin-clouds.pdf")


if __name__ == "__main__":
    clouds()
