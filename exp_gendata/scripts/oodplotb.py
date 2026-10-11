from __future__ import annotations

import json
from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

MODELS = [
    ("Qwen3-8B", "Qwen3-8B", "#2563eb"),
    ("Meta-Llama-3.1-8B-Instruct", "Llama-3.1-8B", "#dc2626"),
    ("gemma-2-9b-it", "gemma-2-9b", "#059669"),
]
CLIP = 0.999


def logdev(a: float) -> float:
    return float(np.log10(1.0 - min(a, CLIP)))


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--field", default="lang")
    parser.add_argument("--refN", type=int, default=64)
    args = parser.parse_args()
    onto = {
        int(p["pair_id"]): p["concept"] for p in json.loads(Path("data/semantic.ontology.json").read_text())["pairs"]
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, (tag, label, color) in zip(axes, MODELS):
        p = Path(f"work/oodcurve/oodcurveb-{tag}.json")
        if not p.exists():
            ax.set_title(f"{label}: (missing)")
            continue
        s = json.loads(p.read_text()).get(args.field)
        pc = json.loads(Path(f"work/oodcurve/oodcurveb-{tag}-percpt.json").read_text()).get(args.field, [])
        if not s:
            continue
        band = s["best_band"]
        N = [b["N"] for b in band]
        med = [logdev(b["median"]) for b in band]
        lo = [logdev(b["p97_5"]) for b in band]
        hi = [logdev(b["p2_5"]) for b in band]
        ax.plot(N, med, "-o", color=color, lw=2, ms=4)
        ax.fill_between(N, lo, hi, color=color, alpha=0.15)
        key = str(args.refN)
        avail = [(c["pid"], c["curve"][key]) for c in pc if key in c["curve"]]
        if avail:
            fast = max(avail, key=lambda t: t[1])
            slow = min(avail, key=lambda t: t[1])
            for pid, a, tag2 in [(fast[0], fast[1], "fastest"), (slow[0], slow[1], "slowest")]:
                ax.annotate(
                    f"{onto.get(pid, pid)}\n({tag2}, AUROC {a:.2f})",
                    xy=(args.refN, logdev(a)),
                    xytext=(args.refN * 0.5, logdev(a) + (0.4 if tag2 == "slowest" else -0.4)),
                    fontsize=7,
                    ha="center",
                    arrowprops={"arrowstyle": "->", "color": "gray", "lw": 0.8},
                )
        cr = s.get("crossing_spearman_N8_vs_N64", float("nan"))
        ax.set_title(
            f"{label}  best L{s['best_layer']}\nN*={s['nstar']['median']} "
            f"(p10–p90 {s['nstar']['p10']}–{s['nstar']['p90']}), cross ρ={cr:.2f}",
            fontsize=10,
        )
        ax.set_xscale("log", base=2)
        ax.set_xlabel("stories per class (N)")
        ax.grid(True, which="both", ls=":", alpha=0.4)
    axes[0].set_ylabel("log10(1 − held-out AUROC)   (lower = better)")
    fig.suptitle(
        f"Vector learning curve, per-{args.field} cells — median + 95% band across concepts "
        "(annotations are N-specific: curves cross)",
        fontsize=12,
    )
    fig.tight_layout()
    out = Path(f"work/oodcurve/oodcurveb-{args.field}.svg")
    fig.savefig(out)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    print(f"wrote {out} (+ .png)")


if __name__ == "__main__":
    main()
